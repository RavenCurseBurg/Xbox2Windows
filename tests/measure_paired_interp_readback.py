"""Render the two packed interpolation registers from #2422. Args: renderer adapter dxc fxc llvm-mc."""
from pathlib import Path
import struct
import sys
import tempfile

import startup_gfx2hlsl_interp_f16_check as vertex
import startup_gfx2hlsl_interp_p2_hi_check as fixture
from startup_flat_param_check import assemble, compile_hlsl, draw


def packed(low, high):
    return int.from_bytes(struct.pack('<ee', low, high), 'little')


def words(values):
    return ','.join('%08x' % fixture.f32_bits(value) for value in values)


def verify(renderer, adapter, dxc, fxc, llvm):
    cases = ((1.5, -2., 0.5, 3.), (-0.75, 0.0625, 2.5, -1.25), (0.25, -0.125, -1.5, 4.))
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        for i, (x, y, z, w) in enumerate(cases):
            source = vertex.VS_SOURCE.replace('0xc0003e00', hex(packed(x, y))).replace('0x42003800', hex(packed(z, w)))
            vs_words = assemble(llvm, source)
            if i == 0:
                assert vs_words == fixture.VS_WORDS
            pipe = fixture.G.Pipeline(vs_words, fixture.PS_WORDS, fixture.root_blob())
            ps = pipe.hlsl_ps()
            high_write = 'v8 = (v8 & 0xffffu) | (f32tof16(i.param0.z) << 16);'
            assert ps.count(high_write) == 1
            expected = words((x, z, y, w))
            variants = (
                ('paired', ps, expected),
                ('cleared_low', ps.replace(high_write, 'v8 = f32tof16(i.param0.z) << 16;'), words((0., z, y, w))),
                ('wrote_low', ps.replace(high_write, 'v8 = (v8 & 0xffff0000u) | f32tof16(i.param0.z);'), words((z, 0., y, w))),
            )
            for kind, compiler, profile in (('fxc', fxc, '5_1'), ('dxc', dxc, '6_0')):
                vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(), f'{kind}_{i}_vs', 'vs_' + profile, directory)
                for name, shader, actual in variants:
                    cso = compile_hlsl(compiler, kind, shader, f'{kind}_{i}_{name}', 'ps_' + profile, directory)
                    read = draw(renderer, adapter, vs, cso, expected)
                    if name == 'paired':
                        assert read.returncode == 0, (name, read.stdout, read.stderr)
                    else:
                        assert actual != expected and read.returncode != 0, (name, read.stdout, read.stderr)
                    assert 'words=' + actual in read.stdout, (name, read.stdout, read.stderr)
                    print(kind, i, name, actual, flush=True)


if __name__ == '__main__':
    verify(*sys.argv[1:])
