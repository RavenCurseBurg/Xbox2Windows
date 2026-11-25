"""Read the POSITION and TEXCOORD streams through a gfx1013 vertex descriptor pointer (#2069).

binary <llvm-mc> checks the three assembled programs. For device readback use
draw <startup_stream_consumer> <warp|hardware> <fxc|dxc> <compiler>.
"""
import os
from pathlib import Path
import struct
import sys
import tempfile

from startup_flat_param_check import assemble
from startup_mrt_check import compile_shader
from startup_stream_check import container
from startup_stream_fetch_check import FETCHED, SWAPPED, Run, first_difference, gfx2hlsl


BODY = """s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v0, s2, v0
buffer_load_dwordx3 v[4:6], v0, s[16:19], 0 idxen
s_waitcnt vmcnt(0)
v_mov_b32 v7, 1.0
exp pos0 v4, v5, v6, v7 done
s_endpgm
"""
BODY_WORDS = '7fc08cbf 0200004a 00203ce000040480 703f8cbf f2020e7e cf0800f804050607 000081bf'
PROGRAMS = (
    ('direct', 's_load_dwordx4 s[16:19], s[0:1], 16\n', '000408f4100000fa'),
    ('out-of-place', 's_add_u32 s24, s0, 16\ns_addc_u32 s25, s1, 0\ns_load_dwordx4 s[16:19], s[24:25], 0\n',
     '00901880 01801982 0c0408f4000000fa'),
    ('in-place', 's_add_u32 s0, s0, 16\ns_addc_u32 s1, s1, 0\ns_load_dwordx4 s[16:19], s[0:1], 0\n',
     '00900080 01800182 000408f4000000fa'),
)


def draw(folder, name, program, consumer, backend, kind, compiler):
    run = Run(folder / name, consumer, backend, container(1, [w for w, in struct.iter_unpack('<I', program)]), False)
    status = gfx2hlsl.translate_request(str(run.cache), run.key)
    assert status == 'ok', status
    stem = run.cache / f'{run.key:016x}'
    manifest = stem.with_suffix('.txt')
    original = manifest.read_text()
    expected_srv = 'srv 0 buf vb 1'
    assert [line for line in original.splitlines() if line.startswith('srv ')] == [expected_srv], original
    for stage in ('vs', 'ps'):
        compile_shader(compiler, kind, stage, stem.with_suffix(f'.{stage}.hlsl'), stem.with_suffix(f'.{stage}.cso'),
                       folder / f'{stage}.asm')
    # Both slots contain float3 positions for two different shapes. Keep the compiled shaders,
    # both vertex views and both draw calls fixed, and vary the bound slot in the manifest.
    for slot, reference in ((1, FETCHED), (0, SWAPPED)):
        manifest.write_text(original.replace(expected_srv, f'srv 0 buf vb {slot}'), encoding='ascii')
        result = run.draw()
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0 and '-> drawn;' in result.stderr, result.stdout + result.stderr
        actual = run.output.read_bytes()
        assert actual == reference, first_difference(actual, reference)
        changed = sum(actual[i:i+4] != FETCHED[i:i+4] for i in range(0, len(actual), 4))
        assert changed == (0 if slot == 1 else 11), changed
        print(f'{kind} {backend} {name} slot {slot}: 32 pixels match; {changed} differ from POSITION')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == 'binary':
        for name, prefix, words in PROGRAMS:
            assert assemble(sys.argv[2], prefix + BODY) == bytes.fromhex(words + BODY_WORDS), name
        print('All three vertex descriptor programs match gfx1013 assembly')
    elif len(sys.argv) == 6 and sys.argv[1] == 'draw' and sys.argv[4] in ('fxc', 'dxc'):
        os.environ['X2W_FXC'] = ''
        with tempfile.TemporaryDirectory() as tmp:
            for name, _prefix, words in PROGRAMS:
                draw(Path(tmp), name, bytes.fromhex(words + BODY_WORDS), *sys.argv[2:])
    else:
        raise SystemExit(__doc__)
