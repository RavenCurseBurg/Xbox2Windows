"""Read back scalar bitset, bit clear and first-set-bit pixel programs on Windows.

Usage: startup_sop1_device_probe.py <LLVM-15-bin> <dxc> <offscreen_check> <warp|hardware>
"""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from startup_image_3d_check import G, FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob
from compute_store_check import clang_assemble

MASK = 0xffffffff


def make_pixel(tmp, llvm, commands):
    # Publish both halves of s4 and s5 as exactly representable floats.
    program = [*commands, "v_mov_b32_e32 v0, s4", "v_mov_b32_e32 v2, s5"]
    for v in (0, 2):
        program += [f"v_lshrrev_b32_e32 v{v+1}, 16, v{v}", f"v_and_b32_e32 v{v}, 0xffff, v{v}"]
        for lane in (v, v+1):
            program.append(f"v_cvt_f32_u32_e32 v{lane}, v{lane}")
    program += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    raw = clang_assemble(tmp, str(llvm / "clang.exe"), str(llvm / "llvm-objcopy.exe"), program)
    return G.Pipeline(VS_WORDS, raw, root_blob()).hlsl_ps()


def cases():
    for index in (0, 31, 32, 37, 0xffffffff):
        bit = 1 << (index & 31)
        old_a, old_b = 0x255ac30c, 0xe9d2fbe1
        yield "set-clear-" + str(index), [
            f"s_mov_b32 s2, {index:#x}", f"s_mov_b32 s4, {old_a:#x}", f"s_mov_b32 s5, {old_b:#x}",
            "s_bitset1_b32 s4, s2", "s_bitset0_b32 s5, s2"], (old_a | bit, old_b & ~bit)
    for value in (0, 1, 0x80, 0x80000000, 0x80004000, 0xffffffff):
        first = (value & -value).bit_length() - 1
        yield "ff1-" + str(value), [
            f"s_mov_b32 s4, {value:#x}", "s_cmp_eq_u32 1, 1", "s_ff1_i32_b32 s4, s4",
            "s_cselect_b32 s5, 1, 0"], (first & MASK, 1)
    # Index and destination name the same SGPR; the source uses the old value.
    yield "alias", ["s_mov_b32 s4, 5", "s_mov_b32 s5, 0x25", "s_bitset1_b32 s4, s4",
                    "s_bitset0_b32 s5, s5"], (0x25, 5)
    for expected in (0, 1):
        yield f"scc-{expected}", ["s_mov_b32 s4, 9", f"s_cmp_eq_u32 1, {expected}",
                                  "s_bitset1_b32 s4, 1", "s_bitset0_b32 s4, 3",
                                  "s_cselect_b32 s5, 1, 0"], (3, expected)


def verify(llvm, dxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        vs = compile_hlsl(dxc, "dxc", FULLSCREEN_VS, "full", "vs_6_0", tmp)
        positives = negatives = 0
        for name, commands, (a, b) in cases():
            def check(lines, positive):
                hlsl = make_pixel(tmp, Path(llvm), lines)
                ps = compile_hlsl(dxc, "dxc", hlsl, "pixel", "ps_6_0", tmp)
                case = tmp / "reference.bin"
                case.write_bytes(struct.pack("<4f", a & 0xffff, a >> 16, b & 0xffff, b >> 16) * 16)
                result = subprocess.run([renderer, adapter, "--startup-pixel-words", vs, ps, "4", "4", case],
                                        capture_output=True, text=True)
                if positive:
                    assert result.returncode == 0 and "all 16 pixels matched" in result.stdout, (name, result.stderr)
                else:
                    assert result.returncode == 1 and "pixel mismatch" in result.stderr, (name, result.stderr)
            check(commands, True)
            positives += 1
            if name == "set-clear-0":
                check([x for x in commands if "s_bitset1" not in x], False)
                negatives += 1
            if name == "alias":
                check([x for x in commands if "s_bitset0" not in x], False)
                negatives += 1
            if name == "ff1-0":
                check([x.replace("s_ff1_i32_b32 s4, s4", "s_mov_b32 s4, 0") for x in commands], False)
                negatives += 1
            if name == "scc-0":
                check([x.replace("s_cmp_eq_u32 1, 0", "s_cmp_eq_u32 1, 1") for x in commands], False)
                negatives += 1
            print(name, flush=True)
        print(f"{adapter}: {positives} gfx1013 pixel programs, 16 pixels each, {negatives} negative controls")


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[4] not in ("warp", "hardware"):
        sys.exit(__doc__)
    verify(*sys.argv[1:])
