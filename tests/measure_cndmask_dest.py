"""Exercise SDWA PAD and DWORD destinations via lossless finite-float exports."""
import hashlib
from pathlib import Path
import struct
import sys
import tempfile

import startup_gfx2hlsl_sdwa_cndmask_check as base
from startup_flat_param_check import assemble

ASM = '''\
v_mul_u32_u24_e32 v1, 0x9b5d, v0
v_mov_b32_e32 v2, 0xdeadbeef
v_cmp_gt_u32_e32 vcc_lo, 0x80000000, v1
v_cndmask_b32_sdwa v2, v1, v0, vcc_lo dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_0
v_cndmask_b32_sdwa v3, v0, v1, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
v_and_b32_e32 v4, 0xffff, v2
v_lshrrev_b32_e32 v5, 16, v2
v_and_b32_e32 v6, 0xffff, v3
v_lshrrev_b32_e32 v7, 16, v3
v_or_b32_e32 v4, 1.0, v4
v_or_b32_e32 v5, 1.0, v5
v_or_b32_e32 v6, 1.0, v6
v_or_b32_e32 v7, 1.0, v7
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
'''

VARIANTS = (
    ('reference', ASM),
    ('preserve', ASM.replace('dst_sel:BYTE_1 dst_unused:UNUSED_PAD', 'dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE')),
    ('low-word', ASM.replace('dst_sel:DWORD dst_unused:UNUSED_PAD', 'dst_sel:WORD_0 dst_unused:UNUSED_PAD')),
)


def model(kind):
    result = []
    for address, product, take_second in base.lanes():
        byte = address & 255 if take_second else (product >> 16) & 255
        padded = byte << 8
        if kind == 'preserve':
            padded |= 0xdead00ef
        full = product if take_second else address
        if kind == 'low-word':
            full &= 65535
        # Embed each half-word in the mantissa of 1.0, retaining all bits of both raw results.
        result.extend(base.ONE | half for half in (padded & 0xffff, padded >> 16, full & 0xffff, full >> 16))
    assert len(result) == 1024 and all(base.ONE <= word <= base.ONE + 0xffff for word in result)
    return result


def run(llvm, dxc, fxc, renderer, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        references = {}
        shaders = {}
        for name, asm in VARIANTS:
            words = assemble(llvm, asm)
            pipeline = base.G.Pipeline(base.VS_WORDS, words, base.root_blob(), ps_hw_inputs=base.PACKED_INPUT)
            shaders[name] = pipeline.hlsl_ps()
            references[name] = folder / (name + '.bin')
            references[name].write_bytes(struct.pack('<1024I', *model(name)))
            print('program', name, len(words), hashlib.sha256(words).hexdigest(), flush=True)
        for kind, compiler, profile in [('fxc', fxc, '5_1'), ('dxc', dxc, '6_0')]:
            vs = base.compile_hlsl(compiler, kind, base.FULLSCREEN_VS, 'vs_' + kind, 'vs_' + profile, folder)
            for name, _ in VARIANTS:
                ps = base.compile_hlsl(compiler, kind, shaders[name], name + kind, 'ps_' + profile, folder)
                run = base.run_pixel_words(renderer, adapter, vs, ps, references[name])
                assert run.returncode == 0 and 'all 256 pixels matched' in run.stdout, (name, run.stdout, run.stderr)
                if name == 'reference':
                    print(kind, name, run.stdout, flush=True)
                    continue
                wrong = base.run_pixel_words(renderer, adapter, vs, ps, references['reference'])
                assert wrong.returncode == 1 and 'pixel mismatch' in wrong.stderr, (wrong.stdout, wrong.stderr)
                diff = sum(a != b for a, b in zip(model(name), model('reference')))
                assert diff == {'preserve': 512, 'low-word': 254}[name]
                first = next(line for line in wrong.stderr.splitlines() if 'pixel mismatch' in line)
                print(kind, name, 'changed words', diff, 'first', first, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run(*sys.argv[1:]))
