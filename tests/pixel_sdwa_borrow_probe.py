"""Compare a derived SDWA borrow input with the same bit supplied by the caller (#2731)."""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import pixel_aot_check as pixel


def forced_reference(row, carry, first_word):
    a, b, _, d = row
    sub = (d >> 16) - (b >> 16) - carry
    signed_byte = int.from_bytes(bytes([(a >> 8) & 255]), "little", signed=True)
    rev = (d & 255) - (signed_byte & 0xffffffff) - (sub < 0)
    return first_word, sub & 0xffff, ((rev & 0xffff) << 16) | 0x1337, int(rev < 0)


def probe(dxc, renderer, backend, directory):
    source = pixel.translate_int(pixel.uncompressed_export(pixel.SDWA_INT_EXTRA_PROGRAMS["co_ci"]))
    original = "uint ci26 = cmp25 ? 1u : 0u;"
    assert source.count(original) == 1
    shaders = {}
    for label, hlsl in (("derived", source), ("uniform", source.replace(original, "uint ci26 = params[1].w;"))):
        path, shader = directory / f"{label}.hlsl", directory / f"{label}.cso"
        path.write_text(hlsl, encoding="ascii")
        pixel.compile_shader(dxc, path, shader)
        shaders[label] = shader

    mismatches = 0
    for idx, (row, carry) in enumerate(zip(pixel.SDWA_INT_EXTRA_ROWS["co_ci"], (1, 0, 1), strict=True)):
        expected = pixel.sdwa_int_extra_reference("co_ci", row)
        assert forced_reference(row, carry, expected[0]) == expected
        opposite = forced_reference(row, 1 - carry, expected[0])
        assert opposite != expected
        for label, shader, supplied, want in (("derived", "derived", 0, expected),
                                              ("matched", "uniform", carry, expected),
                                              ("flipped", "uniform", 1 - carry, opposite)):
            constants = bytearray(256)
            struct.pack_into("<5I", constants, 12, *row, supplied)
            case, raw = directory / "draw.case", directory / "draw.bin"
            case.write_bytes(constants + struct.pack("<4I", *want) * 64 + b"\x01")
            raw.unlink(missing_ok=True)
            run = subprocess.run([renderer, backend, "--check-pixel-case", shaders[shader], case, raw],
                                 capture_output=True, text=True)
            if run.returncode == 77:
                print(run.stdout + run.stderr)
                return 77
            if run.returncode not in (0, 1) or not raw.exists() or raw.stat().st_size != 64 * 64 * 16:
                raise RuntimeError(run.stdout + run.stderr)
            if "debug_layer=enabled" in run.stdout and "debug_errors=0 debug_warnings=0" not in run.stdout:
                raise RuntimeError(run.stdout + run.stderr)
            pixels = tuple(struct.iter_unpack("<4I", raw.read_bytes()))
            wrong = sum(words != want for words in pixels)
            if bool(run.returncode) != bool(wrong):
                raise RuntimeError(run.stdout + run.stderr)
            # On a divergent result show the first failing pixel, even when pixel zero matched.
            observed = next((words for words in pixels if words != want), pixels[0])
            def words(values):
                return " ".join(f"{value:08x}" for value in values)
            print(f"row {idx} {label:7s} expected {words(want)}  observed {words(observed)}  "
                  f"mismatched pixels {wrong}/{len(pixels)}")
            mismatches += wrong
    return int(bool(mismatches))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxc", type=Path)
    parser.add_argument("renderer", type=Path)
    parser.add_argument("backend", choices=("hardware", "warp"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        raise SystemExit(probe(args.dxc.resolve(), args.renderer.resolve(), args.backend, Path(tmp)))
