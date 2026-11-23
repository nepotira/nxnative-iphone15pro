"""Unit tests for dm/tools/log_audit.py."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('log_audit', ROOT / 'dm/tools/log_audit.py')
assert spec is not None and spec.loader is not None
log_audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = log_audit
spec.loader.exec_module(log_audit)

SAMPLE_LOG = """boot: env DM_LAUNCHER=0 (env.txt)
[dm] front end: thermal state 0, brightness 0.65, Low Power Mode off, screen 120 Hz
[dm] [mem] guest memory: memory pool 3.21 GB and the host's margin 1.60 GB (half the pool), room to the memory limit 5.99 GB, DM_GUEST_FILE unset: anonymous
[dm] front end: thermal state → 2 (113 s after launch)
[dm] [nvn-mtl] memory up to frame 3000: resident 4700 MB, GPU allocated 2600 MB, deferred-release queue 20, residency set 1600, serial objects 1700, headroom to the system memory limit 1440 MB
[dm] [nvn] timing up to frame 3000: frame time p50 16.65 / p95 17.50 / max 58.26 ms (300 frames); frame work p50 16.64 / p95 17.48 / max 58.25 ms (300 frames); NVN submit+present p50 2.10 / p95 3.10 / max 4.50 ms (300 frames); GPU p50 12.60 / p95 13.80 / max 53.64 ms (300 frames);
[dm] [nvn] on-screen up to frame 3000: 299 frames, dropped 7 (discarded 1), p99 45.84 ms; footprint 4958 MiB; brightness 0.65; thermal 2
[dm] [nvn] power up to frame 3000: CPU energy 0.21 W, 3.6 mJ per frame (58.2 fps); CPU 80% (100 = one core), 19% of it on performance cores; wakeups 409/s (out of platform idle 7); GPU 0.0 ms/s; thermal 2; top yielders: none
[dm] [nvn] residency set up to frame 3000: commits 12 / 0.6 ms, attachments per frame load 54 MB / store 81 MB, load-all store-all passes 16, saved by not storing unwritten attachments 5.5 MB
[dm] [nvn] GPU texture storage up to frame 3000: 901 regions, 3295.9 MB written into their storage on the GPU (3.00 regions, 10.99 MB per frame)
[dm] [nvn] translation up to frame 3000: 27437 instructions per draw, submit per frame 4.31 M instructions / CPU 1.730 ms, 157 draws per frame
[dm] [nvn] translation detail up to frame 3000: submit per frame 3.46 M cycles (0.80 cycles per instruction), CPU 1.730 ms of which kernel 0.118 ms, wall clock 1.733 ms; mean clock 2.00 GHz (cycles / CPU time);
[dm] [longframe] frame 3010 (50.10 s) whole frame 58.3 ms, submit+present 0.7 ms, GPU 53.6 ms; new pipelines 0 + shader libraries 0, 0.0 ms in all; texture uploads 0 first + 0 re-uploads, 0.0 MB / 0.0 ms; draws with a hybrid pipeline 0
[dm] [longframe] frame 3020 (50.30 s) whole frame 48.7 ms, submit+present 40.9 ms, GPU 13.1 ms; new pipelines 0 + shader libraries 0, 3.4 ms in all; texture uploads 2 first + 0 re-uploads, 0.2 MB / 0.1 ms; draws with a hybrid pipeline 14
[dm] oe focus state → 2
[dm] [longframe] frame 3030 (51.00 s) whole frame 102.3 ms, submit+present 1.1 ms, GPU 75.3 ms; new pipelines 0 + shader libraries 0, 0.0 ms in all; texture uploads 0 first + 0 re-uploads, 0.0 MB / 0.0 ms; draws with a hybrid pipeline 0
"""


class LogAuditTests(unittest.TestCase):
    def test_parse_and_classify_longframes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'log.txt'
            p.write_text(SAMPLE_LOG, encoding='utf-8')
            audit = log_audit.parse_log(p)
        self.assertEqual(audit.envs, ['DM_LAUNCHER=0'])
        self.assertAlmostEqual(audit.mem_limit_gb, 5.99)
        self.assertEqual(audit.thermal_transitions, [(2, 113)])
        s = log_audit.summarize(audit, min_frame=1800)
        self.assertEqual(s['dropped_total'], 7)
        self.assertAlmostEqual(s['fps_med'], 58.2)
        self.assertAlmostEqual(s['mj_med'], 3.6)
        self.assertEqual(s['headroom_min_mb'], 1440)
        self.assertAlmostEqual(s['gsb_max_mb'], 10.99)
        self.assertEqual([lf.cause for lf in audit.longframes], ['gpu-spike', 'pipeline/shader', 'system/focus'])


if __name__ == '__main__':
    unittest.main()
