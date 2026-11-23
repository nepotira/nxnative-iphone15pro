"""Regression tests for Darwin ARM64 assembly in dm/host/entry.S."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENTRY_S = ROOT / "dm" / "host" / "entry.S"


class EntryAsmTests(unittest.TestCase):
    def test_no_semicolon_statement_separators(self):
        # On Darwin/Mach-O ARM64, ';' starts a line comment, NOT a statement separator.
        for lineno, raw in enumerate(ENTRY_S.read_text(encoding="utf-8").splitlines(), 1):
            code = raw.split("//", 1)[0]
            self.assertNotIn(
                ";",
                code,
                f"{ENTRY_S}:{lineno} contains ';' outside '//' comment (silently drops trailing instruction on Darwin ARM64): {raw!r}",
            )

    def test_cntpct_jump_table_layout(self):
        text = ENTRY_S.read_text(encoding="utf-8")
        m = re.search(r"Ldm_cntpct_jt:\n(.*?)Ldm_cntpct_jt_end:", text, re.S)
        self.assertIsNotNone(m, "Ldm_cntpct_jt .. Ldm_cntpct_jt_end block missing")
        lines = [ln.split("//", 1)[0].strip() for ln in m.group(1).splitlines()]
        lines = [ln for ln in lines if ln]
        expected = []
        for reg in range(19, 29):
            expected.append(f"mov x{reg}, x0")
            expected.append("b Ldm_cntpct_done")
        self.assertEqual(lines, expected)
        self.assertIn(".if (Ldm_cntpct_jt_end - Ldm_cntpct_jt) != 80", text)


if __name__ == "__main__":
    unittest.main()
