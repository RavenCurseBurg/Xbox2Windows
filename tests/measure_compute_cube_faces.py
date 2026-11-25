"""Dispatch four translated RDNA cube instructions and verify their raw output words (#2463).

    python tests/measure_compute_cube_faces.py binary <llvm-mc>
    python tests/measure_compute_cube_faces.py dispatch <dxc> <offscreen_check> <warp|hardware>

The shader reads a float3 from each input record and stores four raw result words.
The comparison includes all 248 unwritten buffer words. Set XBOX2WINDOWS_ADAPTER to
select a particular hardware adapter with the project's offscreen runner.
"""
import argparse
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import startup_gfx2hlsl_compute_check as fixture


# Each row gives the input vector and the four exact expected results: face, S, T, signed major times two.
ROWS = [
    ((4, 2, -1), (0, 1, -2, 8)),
    ((-4, 2, -1), (1, -1, -2, -8)),
    ((1, 4, -2), (2, 1, -2, 8)),
    ((1, -4, -2), (3, 1, 2, -8)),
    ((1, 2, 4), (4, 1, -2, 8)),
    ((1, 2, -4), (5, -1, -2, -8)),
    ((-4, 4, 2), (2, -4, 2, 8)),
    ((4, 2, -4), (5, -4, -2, -8)),
    ((2, -4, 4), (4, 2, 4, 8)),
    ((-4, 4, -4), (5, 4, -4, -8)),
]
# All magnitudes tie and -0 compares equal to +0, so +Z wins. The signs
# of the S, T and doubled-major results remain visible in the raw UAV.
ROWS += [
    ((x, y, z), (4.0, x, -y, z))
    for x in (0.0, -0.0)
    for y in (0.0, -0.0)
    for z in (0.0, -0.0)
]
SOURCE = """s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0x0
s_waitcnt lgkmcnt(0)
v_lshlrev_b32 v4, 4, v0
buffer_load_dwordx4 v[8:11], v4, s[12:15], 0 offen
s_waitcnt vmcnt(0)
v_cubeid_f32 v12, v8, v9, v10
v_cubesc_f32 v13, v8, v9, v10
v_cubetc_f32 v14, v8, v9, v10
v_cubema_f32 v15, v8, v9, v10
buffer_store_dwordx4 v[12:15], v4, s[16:19], 0 offen
s_endpgm
"""
PROGRAM = bytes.fromhex(
    '06038abe80038bbe05030cf4000000fa7fc08cbf84000834001038e004080380'
    '703f8cbf0c0044d508132a040d0045d508132a040e0046d508132a040f0047d5'
    '08132a04001078e0040c0480000081bf'
)

def mutate(hlsl, name):
    if name in ('correct', 'precise'):
        return hlsl
    if name == 'face':
        old = 'float3 cube0 = float3(F(v8), F(v9), F(v10))'
        assert hlsl.count(old) == 1
        return hlsl.replace(old, 'float3 cube0 = float3(F(v9), F(v8), F(v10))')
    if name == 'ties':
        text, count = re.subn(r'(abs\(cube[0-3]\)\.z) >=', r'\1 >', hlsl)
        assert count == 8
        return text
    reg, op = {'sc': (13, '^= 0x80000000u'), 'tc': (14, '^= 0x80000000u'), 'ma': (15, '&= 0x7fffffffu')}[name]
    text, count = re.subn(rf'(?m)^(    v{reg} = U\(.*;)$', rf'\1\n    v{reg} {op};', hlsl)
    assert count == 1
    return text


def write_case(path):
    inputs = bytearray(fixture.WORDS * 4)
    expected = [fixture.SENTINEL] * fixture.WORDS
    for lane, (direction, results) in enumerate(ROWS):
        struct.pack_into('<4f', inputs, 16 * lane, *direction, 153.0)
        expected[4 * lane:4 * lane + 4] = struct.unpack('<4I', struct.pack('<4f', *results))
    path.write_bytes(struct.pack('<I320I16I', 0x010101, *expected, *([0] * 16)) + inputs)


def dispatch(dxc, runner, adapter, work):
    hlsl = fixture.G.ComputePipeline(PROGRAM, fixture.root_blob(), fixture.USER7, (len(ROWS), 1, 1)).hlsl_cs()
    case = work / 'cube.case'
    write_case(case)
    expected_mismatches = {'correct': 0, 'precise': 0, 'face': 5, 'sc': 18, 'tc': 18, 'ma': 9, 'ties': 25}
    for name, count in expected_mismatches.items():
        text = mutate(hlsl, name)
        src, dxil = work / (name + '.hlsl'), work / (name + '.dxil')
        src.write_text(text, encoding='ascii')
        flags = ['-Gis'] if name == 'precise' else []
        subprocess.run([dxc, *flags, '-T', 'cs_6_0', '-E', 'main', '-Fo', dxil, src], check=True)
        run = subprocess.run([runner, adapter, '--compute-store-case', dxil, case], capture_output=True, text=True)
        log = run.stdout + run.stderr
        print(name, log, flush=True)
        if run.returncode == 77:
            raise SystemExit(77)
        summary = f'groups=1x1x1 compared_words=320 mismatches={count}'
        if run.returncode != int(count != 0) or summary not in log.splitlines():
            raise AssertionError(f'{name}: expected {count} mismatches, exit {run.returncode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('binary').add_argument('llvm_mc')
    gpu = modes.add_parser('dispatch')
    gpu.add_argument('dxc')
    gpu.add_argument('runner')
    gpu.add_argument('adapter', choices=['hardware', 'warp'])
    args = parser.parse_args()
    if args.mode == 'binary':
        assert fixture.assemble(args.llvm_mc, SOURCE) == PROGRAM
        print('80 fixture bytes match gfx1013 assembly')
    else:
        with tempfile.TemporaryDirectory() as tmp:
            dispatch(args.dxc, args.runner, args.adapter, Path(tmp))
