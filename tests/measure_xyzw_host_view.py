"""Measure the exact-fit boundary of the existing R-raw Store4 with sized UAVs."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import startup_compute_store_format_xyzw_check as fixture


def verify(dxc, renderer, adapter, folder):
    program, source, constants, written = fixture.case_inputs("R-raw")
    untouched = [fixture.SENTINEL] * fixture.WORDS
    changed = [i for i, word in enumerate(written) if word != untouched[i]]
    assert changed == [4, 5, 6, 7]
    assert tuple(fixture.PROGRAMS[program][4]) == (1, 1, 1)
    hlsl = fixture.translate(program).hlsl_cs()
    guard = "if (uav_bytes < elem_bytes || addr > uav_bytes - elem_bytes) return;"
    assert hlsl.count(guard) == 1
    for name, text in (("original", hlsl), ("strict", hlsl.replace(guard, guard.replace("addr >", "addr >=")))):
        src, cso = folder / f"{name}.hlsl", folder / f"{name}.cso"
        src.write_text(text, encoding="ascii")
        subprocess.run([dxc, "-Gis", "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
        for words in (7, 8, 9):
            expected = untouched if words == 7 else written
            case = folder / f"{words}.case"
            case.write_bytes(struct.pack("<I", 0x010101) + struct.pack("<320I", *expected)
                             + struct.pack("<16I", *constants) + source)
            result = subprocess.run([renderer, adapter, "--compute-store-sized-case", cso, case, str(words)],
                                    capture_output=True, text=True)
            if result.returncode == 77 and adapter == "hardware":
                print(result.stdout + result.stderr)
                return 77
            wrong = changed if name == "strict" and words == 8 else []
            diagnostics = [line for line in result.stderr.splitlines()
                           if not (adapter == "hardware" and line.startswith("XBOX2WINDOWS_ADAPTER: skipping "))]
            want = ([f"word=4 expected={written[4]:08x} actual={fixture.SENTINEL:08x}",
                     "Compute readback differs from CPU reference"] if wrong else [])
            if (result.returncode != int(bool(wrong)) or diagnostics != want
                    or f"compared_words=320 mismatches={len(wrong)}" not in result.stdout):
                raise AssertionError(result.stdout + result.stderr)
            print(f"{name} {words}-word view: checked 320 words, mismatches {wrong}")
            if wrong:
                # The renderer reports the mismatch count and the first differing word.
                # Compare all 320 words to the sentinel to establish which four changed.
                case.write_bytes(case.read_bytes()[:4] + struct.pack("<320I", *untouched)
                                 + struct.pack("<16I", *constants) + source)
                control = subprocess.run([renderer, adapter, "--compute-store-sized-case", cso, case, str(words)],
                                         capture_output=True, text=True)
                assert control.returncode == 0 and "compared_words=320 mismatches=0" in control.stdout, (
                    control.stdout + control.stderr)
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(verify(*sys.argv[1:], Path(tmp)))
