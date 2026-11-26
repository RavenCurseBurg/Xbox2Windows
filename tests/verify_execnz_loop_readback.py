"""Reproduce the four-lane EXECNZ loop comparison in #2772 using source-assembled input."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import startup_gfx2hlsl_compute_check as fixture


def guest(reopen, tail=''):
    return f"""s_mov_b32 s10, s6
s_mov_b32 s11, 0
s_load_dwordx8 s[12:19], s[10:11], 0
s_waitcnt lgkmcnt(0)
v_lshlrev_b32_e32 v4, 2, v0
v_mov_b32_e32 v3, 0
s_mov_b32 s20, exec_lo
loop:
{'s_mov_b32 exec_lo, s20' if reopen else ''}
v_add_nc_u32_e32 v3, 1, v3
v_cmp_gt_u32_e32 vcc_lo, v0, v3
s_and_b32 exec_lo, exec_lo, vcc_lo
s_cbranch_execnz loop
s_mov_b32 exec_lo, s20
{tail}
buffer_store_dword v3, v4, s[16:19], 0 offen
s_endpgm
"""


def wave(reopen):
    lanes = list(range(4))
    active = lanes
    counters = [0] * len(lanes)
    while active:
        if reopen:
            active = lanes
        for lane in active:
            counters[lane] += 1
        active = [lane for lane in active if lane > counters[lane]]
    return counters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for tool in ('clang', 'objcopy', 'dxc', 'renderer', 'adapter'):
        p.add_argument(tool)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        d = Path(temp)
        for name, reopen, tail in (
                ('monotone', False, ''), ('reopen', True, ''),
                ('live-scc', False, 's_cselect_b32 s21, 63, 0\nv_add_nc_u32_e32 v3, s21, v3'),
                ('live-vcc', False, 'v_cndmask_b32_e32 v3, 0, v3, vcc_lo')):
            asm, obj, raw = d/'guest.s', d/'guest.o', d/'guest.bin'
            asm.write_text(guest(reopen, tail))
            subprocess.run([args.clang, '-cc1as', '-triple', 'amdgcn--amdpal', '-target-cpu', 'gfx1013',
                            '-target-feature', '+wavefrontsize32', '-filetype', 'obj', asm, '-o', obj], check=True)
            subprocess.run([args.objcopy, '-O', 'binary', '--only-section=.text', obj, raw], check=True)
            try:
                hlsl = fixture.translate('D', code=raw.read_bytes(), group=(4, 1, 1)).hlsl_cs()
            except fixture.G.Unsupported as reason:
                print(name, 'refused:', reason)
                if name == 'monotone':
                    raise
                continue
            if name == 'live-vcc':
                raise RuntimeError('VCC-dependent case unexpectedly translated')
            ps, shader = d/'fixture.hlsl', d/'fixture.dxil'
            ps.write_text(hlsl)
            subprocess.run([args.dxc, '-E', 'main', '-T', 'cs_6_0', '-Fo', shader, ps], check=True)
            for label, first in [('wave', wave(reopen)), ('independent-lane', wave(False))]:
                expected = first + [fixture.SENTINEL] * (fixture.WORDS - 4)
                case = d/'fixture.case'
                case.write_bytes(struct.pack('<I320I16I', 0x010101, *expected, fixture.KEY, fixture.STRIDE, *([0]*14)))
                run = subprocess.run([args.renderer, args.adapter, '--compute-store-case', shader, case],
                                     capture_output=True, text=True)
                print(name, label, first, 'exit', run.returncode, run.stdout.strip(), run.stderr.strip())
                if (name in ('monotone', 'live-scc') or label == 'independent-lane') and run.returncode:
                    raise RuntimeError(f'{name} {label} control failed')
                if run.returncode not in (0, 1):
                    raise RuntimeError('renderer did not complete')


if __name__ == '__main__':
    main()
