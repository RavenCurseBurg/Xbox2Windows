"""Independent 320-word D3D12 reference for the second-operand alias in #2778."""
import argparse
import itertools
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import startup_gfx2hlsl_compute_check as fixture


OPS = ('s_add_u32', 's_sub_u32', 's_add_i32', 's_sub_i32',
       's_min_u32', 's_max_u32', 's_min_i32', 's_max_i32')
MASK = 0xffffffff


def guest():
    lines = ['s_mov_b32 s10, s6', 's_mov_b32 s11, 0', 's_load_dwordx8 s[12:19], s[10:11], 0',
             's_waitcnt lgkmcnt(0)', 'v_mov_b32_e32 v4, 0']
    for op in OPS:
        lines += ['s_mov_b32 s21, s5', f'{op} s21, s4, s21', 's_cselect_b32 s22, 1, 0']
        for register in ('s21', 's22'):
            lines += [f'v_mov_b32_e32 v3, {register}', 'buffer_store_dword v3, v4, s[16:19], 0 offen',
                      'v_add_nc_u32_e32 v4, 4, v4']
    return '\n'.join([*lines, 's_endpgm', ''])


def reference(a, b):
    signed_a, signed_b = struct.unpack('<ii', struct.pack('<II', a, b))
    output = []
    for op in OPS:
        x, y = (signed_a, signed_b) if op.endswith('i32') else (a, b)
        if 'min' in op or 'max' in op:
            value = min(x, y) if 'min' in op else max(x, y)
            flag = value == x
        else:
            value = x + y if 'add' in op else x - y
            low, high = (-(1 << 31), (1 << 31) - 1) if op.endswith('i32') else (0, MASK)
            flag = not low <= value <= high
        output.extend((value & MASK, int(flag)))
    return output + [fixture.SENTINEL] * (fixture.WORDS - len(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('clang', 'objcopy', 'dxc', 'renderer', 'adapter'):
        p.add_argument(key)
    args = p.parse_args()
    values = (0, 1, MASK, 0x7fffffff, 0x80000000, 0x55555555)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        asm, obj, raw = (folder / name for name in ('guest.s', 'guest.o', 'guest.bin'))
        asm.write_text(guest())
        subprocess.run([args.clang, '-cc1as', '-triple', 'amdgcn--amdpal', '-target-cpu', 'gfx1013',
                        '-target-feature', '+wavefrontsize32', '-filetype', 'obj', asm, '-o', obj], check=True)
        subprocess.run([args.objcopy, '-O', 'binary', '--only-section=.text', obj, raw], check=True)
        source = fixture.translate('S', code=raw.read_bytes()).hlsl_cs()
        # Move carry calculation after the aliased write. Its second source then contains the sum.
        lines = source.splitlines()
        positions = [i for i, line in enumerate(lines[:-1])
                     if 'scc =' in line and '<' in line and 's21 =' in lines[i+1] and ' + ' in lines[i+1]]
        assert len(positions) == 1, [line.strip() for line in lines if 'scc =' in line or 's21 =' in line]
        index, = positions
        lines[index:index+2] = reversed(lines[index:index+2])
        for label, shader_text in [('original', source), ('late-carry', '\n'.join(lines))]:
            shader, dxil = folder/'variant.hlsl', folder/'variant.dxil'
            shader.write_text(shader_text)
            subprocess.run([args.dxc, '-T', 'cs_6_0', '-E', 'main', '-Fo', dxil, shader], check=True)
            caught = 0
            for a, b in itertools.product(values, repeat=2):
                case = folder/'input.case'
                case.write_bytes(struct.pack('<I320I16I', 0x010101, *reference(a, b), a, b, *([0]*14)))
                run = subprocess.run([args.renderer, args.adapter, '--compute-store-case', dxil, case],
                                     capture_output=True, text=True)
                wrong = label == 'late-carry' and (a + b > MASK) != (a + ((a + b) & MASK) > MASK)
                if wrong:
                    assert run.returncode == 1 and re.search(r'word=1\b', run.stderr+run.stdout), (a,b,run.stderr,run.stdout)
                    caught += 1
                else:
                    assert run.returncode == 0 and 'compared_words=320 mismatches=0' in run.stdout, (a,b,run.stderr,run.stdout)
            if label == 'late-carry':
                assert caught > 0
            print(label, len(values)**2, 'complete readbacks;', caught, 'predicted word-1 mismatches')


if __name__ == '__main__':
    main()
