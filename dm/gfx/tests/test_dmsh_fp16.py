"""Native Apple AGX half2 SIMD folding for scalarized FP16x2 unpack-op-pack chains (dmsh_rewrite.fp16_pack_chains)."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import dmsh_rewrite   # noqa: E402

HEAD = '#include <metal_stdlib>\n#include <simd/simd.h>\n\nusing namespace metal;\n\n'

HMUL2 = HEAD + '''kernel void main0(device uint* buf [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    uint _10 = buf[i];
    uint _11 = buf[i + 1u];
    float _20 = float2(as_type<half2>(_10)).x;
    float _21 = float2(as_type<half2>(_10)).y;
    float _22 = float2(as_type<half2>(_11)).x;
    float _23 = float2(as_type<half2>(_11)).y;
    float _30 = _20 * _22;
    float _31 = _21 * _23;
    uint _40 = as_type<uint>(half2(float2(_30, _31)));
    buf[i + 2u] = _40;
}
'''

HFMA2_CONST = HEAD + '''kernel void main0(device uint* buf [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    uint _10 = buf[i];
    uint _11 = buf[i + 1u];
    float _20 = float2(as_type<half2>(uint(_10))).x;
    float _21 = float2(as_type<half2>(uint(_10))).y;
    float _22 = float2(as_type<half2>(_11)).x;
    float _23 = float2(as_type<half2>(_11)).y;
    float _30 = fma(_20, _22, 0.5);
    float _31 = fma(_21, _23, 1.0);
    uint _40 = as_type<uint>(half2(float2(_30, _31)));
    buf[i + 2u] = _40;
}
'''

MUTATED_SRC = HEAD + '''kernel void main0(device uint* buf [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    uint _10 = buf[i];
    uint _11 = buf[i + 1u];
    float _20 = float2(as_type<half2>(_10)).x;
    float _21 = float2(as_type<half2>(_10)).y;
    float _22 = float2(as_type<half2>(_11)).x;
    float _23 = float2(as_type<half2>(_11)).y;
    float _30 = _20 + _22;
    float _31 = _21 + _23;
    _10 = 0u;
    uint _40 = as_type<uint>(half2(float2(_30, _31)));
    buf[i + 2u] = _40 + _10;
}
'''

SHARED_LANE = HEAD + '''kernel void main0(device uint* buf [[buffer(0)]], uint i [[thread_position_in_grid]])
{
    uint _10 = buf[i];
    uint _11 = buf[i + 1u];
    float _20 = float2(as_type<half2>(_10)).x;
    float _21 = float2(as_type<half2>(_10)).y;
    float _22 = float2(as_type<half2>(_11)).x;
    float _23 = float2(as_type<half2>(_11)).y;
    float _30 = _20 + _22;
    float _31 = _21 + _23;
    uint _40 = as_type<uint>(half2(float2(_30, _31)));
    buf[i + 2u] = _40 + as_type<uint>(_20);
}
'''


class Fp16PackChains(unittest.TestCase):
    def test_hmul2_collapses_to_half2(self):
        out = dmsh_rewrite.fp16_pack_chains(HMUL2)
        self.assertIn('uint _40 = as_type<uint>((as_type<half2>(_10) * as_type<half2>(_11)));', out)
        self.assertNotIn('float _20', out)
        self.assertNotIn('float _30', out)

    def test_hfma2_with_literal_collapses(self):
        out = dmsh_rewrite.fp16_pack_chains(HFMA2_CONST)
        self.assertIn('uint _40 = as_type<uint>(fma(as_type<half2>(uint(_10)), as_type<half2>(_11), half2(0.5, 1.0)));', out)
        self.assertNotIn('float _30', out)

    def test_mutated_source_is_not_folded(self):
        self.assertEqual(dmsh_rewrite.fp16_pack_chains(MUTATED_SRC), MUTATED_SRC)
        for mut in ('    _10 <<= 1u;', '    _10 >>= 1u;', '    _10 %= 3u;', '    _10++;', '    --_10;'):
            src = MUTATED_SRC.replace('    _10 = _10 + 1u;', mut)
            self.assertEqual(dmsh_rewrite.fp16_pack_chains(src), src, f'should not fold across {mut.strip()}')

    def test_shared_unpack_keeps_live_lane(self):
        out = dmsh_rewrite.fp16_pack_chains(SHARED_LANE)
        self.assertIn('uint _40 = as_type<uint>((as_type<half2>(_10) + as_type<half2>(_11)));', out)
        self.assertIn('float _20 =', out)
        self.assertNotIn('float _21 =', out)
        self.assertNotIn('float _30 =', out)

    @unittest.skipUnless(shutil.which('xcrun') and sys.platform == 'darwin', 'Metal compiler')
    def test_compiles_with_metal(self):
        with tempfile.TemporaryDirectory() as d:
            for name, src in (('hmul2', dmsh_rewrite.fp16_pack_chains(HMUL2)),
                              ('hfma2', dmsh_rewrite.fp16_pack_chains(HFMA2_CONST)),
                              ('shared', dmsh_rewrite.fp16_pack_chains(SHARED_LANE))):
                m = os.path.join(d, name + '.metal')
                Path(m).write_text(src)
                r = subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-c', '-O1', '-fpreserve-invariance',
                                    f'-fmodules-cache-path={d}', m, '-o', os.path.join(d, name + '.air')],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
