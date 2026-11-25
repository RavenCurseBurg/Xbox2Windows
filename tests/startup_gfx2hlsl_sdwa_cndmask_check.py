"""The SDWA form of v_cndmask_b32 in tools/startup/gfx2hlsl.py, drawn on a GPU.

  (no arguments)                       fixture and model controls, pure Python
  binary <llvm-mc>                     assemble the fixture with LLVM 15 (-mcpu=gfx1013) and require its words
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the 16 x 16 pixel fixture (FXC ps_5_1 when given, and DXC ps_6_0),
                                       draw and compare all 1,024 exported words with the model, then require
                                       each mutated shader to fail against the model and match its own

The VOP2 select in SDWA form reads its lane mask from VCC, printed as the fourth operand. Each source takes its
src0_sel/src1_sel field, zero-extended or sign-extended under sext(), the select picks src1 where the lane's VCC
bit is set and src0 elsewhere, and dst_sel/dst_unused place the low bits of the result. These are the integer SDWA
rules the other forms already use (sdwa_source/sdwa_dest); the draw shows a desktop GPU computing the translated
select, not that the console does.

Both fixtures export normal binary32 values to R32G32B32A32_FLOAT. The first preserves the exponent of 1.0
under UNUSED_PRESERVE. The second exports the low and high halves of each select result in the mantissas of
separate channels so that every bit of a padded byte and a full dword can be compared.
"""
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools'))
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble  # noqa: E402
from startup_image_3d_check import FULLSCREEN_VS, VS_WORDS, compile_hlsl, root_blob  # noqa: E402

failures = []

WIDTH = HEIGHT = 16
MULTIPLIER = 0x9B5D
ONE = 0x3F800000
PACKED_INPUT = (0x8000, 0x8000)  # SPI_PS_INPUT_ENA/ADDR: packed screen position x | (y << 16) in v0

SEL = {'BYTE_0': 0, 'BYTE_1': 1, 'BYTE_2': 2, 'BYTE_3': 3, 'WORD_0': 4, 'WORD_1': 5, 'DWORD': 6}
UNUSED = {'UNUSED_PAD': 0, 'UNUSED_SEXT': 1, 'UNUSED_PRESERVE': 2}
VOP2_CNDMASK_B32 = 1


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print(detail)


def encode_cndmask_sdwa(vdst, src0, src1, *, src0_sel, src1_sel, dst_sel, dst_unused='UNUSED_PRESERVE',
                        src0_sext=0, src1_sext=0):
    """VOP2 SDWA (GFX10 VOP_SDWA9Ae): w0 [30:25] opcode, [24:17] vdst, [16:9] VGPR src1, [8:0] = 249;
    w1 [7:0] VGPR src0, [10:8] dst_sel, [12:11] dst_unused, [18:16] src0_sel, [19] src0_sext,
    [26:24] src1_sel, [27] src1_sext."""
    w0 = (VOP2_CNDMASK_B32 << 25) | (vdst << 17) | (src1 << 9) | 249
    w1 = (src0 | (SEL[dst_sel] << 8) | (UNUSED[dst_unused] << 11) | (SEL[src0_sel] << 16) | (src0_sext << 19)
          | (SEL[src1_sel] << 24) | (src1_sext << 27))
    return struct.pack('<II', w0, w1)


# One select per exported channel: (vdst, src0, src0 sext, src0_sel, src1, src1 sext, src1_sel, dst_sel).
SELECTS = (
    (2, 0, 0, 'WORD_0', 1, 0, 'BYTE_1', 'WORD_0'),
    (3, 1, 1, 'BYTE_2', 0, 0, 'WORD_1', 'WORD_0'),
    (4, 1, 0, 'BYTE_3', 1, 0, 'WORD_0', 'BYTE_1'),
    (5, 1, 0, 'BYTE_1', 0, 0, 'BYTE_0', 'BYTE_2'),
)


def assembly(select):
    vdst, s0, x0, sel0, s1, x1, sel1, dst = select
    a = 'sext(v%d)' % s0 if x0 else 'v%d' % s0
    b = 'sext(v%d)' % s1 if x1 else 'v%d' % s1
    return ('v_cndmask_b32_sdwa v%d, %s, %s, vcc_lo dst_sel:%s dst_unused:UNUSED_PRESERVE src0_sel:%s src1_sel:%s'
            % (vdst, a, b, dst, sel0, sel1))


# Constructed gfx1013 pixel stage (LLVM 15 assembles each line to these words; see `binary`):
#   v_mul_u32_u24_e32 v1, 0x9b5d, v0
#   v_mov_b32_e32 v2..v5, 1.0
#   v_cmp_gt_u32_e32 vcc_lo, 0x80000000, v1      (VCC: bit 31 of v1 clear)
#   the four selects above
#   exp mrt0 v2, v3, v4, v5 done vm
#   s_endpgm
PRODUCT = ('v_mul_u32_u24_e32 v1, 0x9b5d, v0', struct.pack('<II', (0x0B << 25) | (1 << 17) | 255, MULTIPLIER))
COMPARE = ('v_cmp_gt_u32_e32 vcc_lo, 0x80000000, v1', struct.pack('<II', 0x7C000000 | (0xC4 << 17) | (1 << 9) | 255,
                                                                  0x80000000))
PROGRAM = (
    [PRODUCT]
    + [('v_mov_b32_e32 v%d, 1.0' % v, struct.pack('<I', 0x7E000000 | (v << 17) | (1 << 9) | 242)) for v in (2, 3, 4, 5)]
    + [COMPARE]
    + [(assembly(s), encode_cndmask_sdwa(s[0], s[1], s[4], src0_sel=s[3], src1_sel=s[6], dst_sel=s[7],
                                         src0_sext=s[2], src1_sext=s[5])) for s in SELECTS]
    + [('exp mrt0 v2, v3, v4, v5 done vm', bytes.fromhex('0f1800f802030405')),
       ('s_endpgm', bytes.fromhex('000081bf'))]
)
PS_WORDS = b''.join(raw for _, raw in PROGRAM)

# The model of the hardware, and the wrong variants each mutation implements.
MODES = ('swapped operands', 'mask ignored', 'mask inverted', 'dropped sext', 'neighbouring byte',
         'wrong destination byte')
DESTINATION_MODES = ('preserved padding', 'low word only')


def destination_program(mode):
    unused = 'UNUSED_PRESERVE' if mode == 'preserved padding' else 'UNUSED_PAD'
    width = 'WORD_0' if mode == 'low word only' else 'DWORD'
    return [
        PRODUCT, ('v_mov_b32_e32 v2, 0xdeadbeef', bytes.fromhex('ff02047e efbeadde')), COMPARE,
        (f'v_cndmask_b32_sdwa v2, v1, v0, vcc_lo dst_sel:BYTE_1 dst_unused:{unused} src0_sel:BYTE_2 src1_sel:BYTE_0',
         encode_cndmask_sdwa(2, 1, 0, src0_sel='BYTE_2', src1_sel='BYTE_0', dst_sel='BYTE_1', dst_unused=unused)),
        (f'v_cndmask_b32_sdwa v3, v0, v1, vcc_lo dst_sel:{width} dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD',
         encode_cndmask_sdwa(3, 0, 1, src0_sel='DWORD', src1_sel='DWORD', dst_sel=width, dst_unused='UNUSED_PAD')),
        ('v_and_b32_e32 v4, 0xffff, v2', bytes.fromhex('ff040836 ffff0000')),
        ('v_lshrrev_b32_e32 v5, 16, v2', bytes.fromhex('90040a2c')),
        ('v_and_b32_e32 v6, 0xffff, v3', bytes.fromhex('ff060c36 ffff0000')),
        ('v_lshrrev_b32_e32 v7, 16, v3', bytes.fromhex('90060e2c')),
        ('v_or_b32_e32 v4, 1.0, v4', bytes.fromhex('f2080838')),
        ('v_or_b32_e32 v5, 1.0, v5', bytes.fromhex('f20a0a38')),
        ('v_or_b32_e32 v6, 1.0, v6', bytes.fromhex('f20c0c38')),
        ('v_or_b32_e32 v7, 1.0, v7', bytes.fromhex('f20e0e38')),
        ('exp mrt0 v4, v5, v6, v7 done vm', bytes.fromhex('0f1800f8 04050607')),
        ('s_endpgm', bytes.fromhex('000081bf')),
    ]


def destination_hlsl(mode):
    words = b''.join(raw for _, raw in destination_program(mode))
    return G.Pipeline(VS_WORDS, words, root_blob(), ps_hw_inputs=PACKED_INPUT).hlsl_ps()


def field(value, sel, sext):
    shift, width = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
                    'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}[sel]
    part = (value >> shift) & ((1 << width) - 1)
    if sext and width < 32 and part >> (width - 1):
        part |= (0xFFFFFFFF << width) & 0xFFFFFFFF
    return part


def place(old, result, sel, preserve):
    shift, width = {'BYTE_0': (0, 8), 'BYTE_1': (8, 8), 'BYTE_2': (16, 8), 'BYTE_3': (24, 8),
                    'WORD_0': (0, 16), 'WORD_1': (16, 16), 'DWORD': (0, 32)}[sel]
    mask = ((1 << width) - 1) << shift
    return ((old & ~mask) if preserve else 0) & 0xFFFFFFFF | ((result << shift) & mask)


def lanes():
    for y in range(HEIGHT):
        for x in range(WIDTH):
            v0 = x | (y << 16)
            v1 = (v0 & 0xFFFFFF) * MULTIPLIER & 0xFFFFFFFF
            yield v0, v1, 0x80000000 > v1


def expected_pixel_words(mode='reference'):
    words = []
    for v0, v1, vcc in lanes():
        regs = {0: v0, 1: v1}
        for channel, (_, s0, x0, sel0, s1, x1, sel1, dst) in enumerate(SELECTS):
            if mode == 'dropped sext' and channel == 1:
                x0 = 0
            if mode == 'neighbouring byte' and channel == 3:
                sel0 = 'BYTE_2'
            if mode == 'wrong destination byte' and channel == 2:
                dst = 'BYTE_0'
            a, b = field(regs[s0], sel0, x0), field(regs[s1], sel1, x1)
            if mode == 'swapped operands' and channel == 0:
                a, b = b, a
            lane = (not vcc) if (mode == 'mask inverted' and channel == 2) else vcc
            picked = a if (mode == 'mask ignored' and channel == 3) else (b if lane else a)
            words.append(place(ONE, picked, dst, True))
    return words


def normal_or_zero(word):
    exponent = (word >> 23) & 0xFF
    return word & 0x7FFFFFFF == 0 or 0 < exponent < 0xFF


def destination_pixel_words(mode):
    words = []
    for position, product, vcc in lanes():
        selected_byte = position & 0xff if vcc else (product >> 16) & 0xff
        padded = selected_byte << 8
        if mode == 'preserved padding':
            padded |= 0xdead00ef
        full = product if vcc else position
        if mode == 'low word only':
            full &= 0xffff
        words.extend(ONE | half for half in (padded & 0xffff, padded >> 16, full & 0xffff, full >> 16))
    return words


def pixel_hlsl(mode='reference'):
    text = G.Pipeline(VS_WORDS, PS_WORDS, root_blob(), ps_hw_inputs=PACKED_INPUT).hlsl_ps()
    if mode == 'reference':
        return text
    # Each mutation edits one select statement of the translated shader.
    lines = text.split('\n')
    targets = {'swapped operands': 'v2', 'dropped sext': 'v3', 'mask inverted': 'v4', 'wrong destination byte': 'v4',
               'neighbouring byte': 'v5', 'mask ignored': 'v5'}
    register = targets[mode]
    found = [i for i, line in enumerate(lines) if re.match(r'\s*(uint\s+)?%s\s*=.*\?' % register, line)]
    assert len(found) == 1, (mode, found, text)
    i = found[0]
    line = lines[i]
    edited = {
        'swapped operands': lambda s: re.sub(r'\? (\(\(v1 >> 8u\) & 0xffu\)) : (\(\(v0 >> 0u\) & 0xffffu\))',
                                             r'? \2 : \1', s),
        'mask inverted': lambda s: re.sub(r'\((\w+) & 1\) \? ', r'((\1 & 1) ^ 1) ? ', s),
        'dropped sext': lambda s: s.replace('asuint(asint(v1 << 8u) >> 24u)', '((v1 >> 16u) & 0xffu)'),
        'wrong destination byte': lambda s: s.replace('& 0xffff00ffu', '& 0xffffff00u').replace(
            '& 0xffu) << 8u)', '& 0xffu) << 0u)'),
        'neighbouring byte': lambda s: s.replace(': ((v1 >> 8u) & 0xffu)', ': ((v1 >> 16u) & 0xffu)'),
        'mask ignored': lambda s: re.sub(r'\(\w+ & 1\) \? ', '(0u & 1) ? ', s),
    }[mode](line)
    assert edited != line, (mode, line)
    lines[i] = edited
    return '\n'.join(lines)


def controls():
    states = [vcc for _, _, vcc in lanes()]
    check('both VCC states occur in at least 64 pixels each', min(states.count(True), states.count(False)) >= 64,
          'set in %d of %d pixels' % (states.count(True), len(states)))
    reference = expected_pixel_words()
    check('every reference word is a normal binary32 value', all(normal_or_zero(w) for w in reference))
    destination = destination_pixel_words('reference')
    for mode, expected_changes in (('reference', 0), ('preserved padding', 512), ('low word only', 254)):
        words = destination_pixel_words(mode)
        check(f'destination ({mode}) exports 1,024 normal words',
              len(words) == 1024 and all(ONE <= w <= ONE + 0xffff for w in words))
        check(f'destination ({mode}) differs in {expected_changes} words',
              sum(a != b for a, b in zip(destination, words)) == expected_changes)
        destination_hlsl(mode)
    for mode in MODES:
        model = expected_pixel_words(mode)
        check(f'model ({mode}) differs from the reference', model != reference)
        check(f'model ({mode}) words are normal binary32 values', all(normal_or_zero(w) for w in model))
    # Every channel of every mutated model differs in at least one pixel of its own channel only.
    for mode in MODES:
        model = expected_pixel_words(mode)
        channels = {i % 4 for i, (a, b) in enumerate(zip(reference, model)) if a != b}
        check(f'model ({mode}) changes one channel', len(channels) == 1, str(channels))
    try:
        text = pixel_hlsl()
    except G.Unsupported as error:
        check('the fixture translates', False, str(error))
        return
    check('the fixture translates', True)
    print(text)
    for mode in MODES:
        try:
            pixel_hlsl(mode)
            check(f'mutation ({mode}) applies to the translated shader', True)
        except AssertionError as error:
            check(f'mutation ({mode}) applies to the translated shader', False, repr(error))


def check_binary(llvm_mc):
    source = '\n'.join(text for text, _ in PROGRAM) + '\n'
    proc = subprocess.run([llvm_mc, '-arch=amdgcn', '-mcpu=gfx1013', '-show-encoding'],
                          input=source, capture_output=True, text=True, check=True)
    built = bytearray()
    for line in proc.stdout.splitlines():
        m = re.search(r'encoding: \[([^\]]*)\]', line)
        if m:
            built += bytes(int(b, 16) for b in m.group(1).split(','))
    check('the fixture words match llvm-mc', bytes(built) == PS_WORDS,
          'llvm-mc %s\nfixture %s' % (bytes(built).hex(), PS_WORDS.hex()))
    printed = [line.split(';')[0].strip() for line in proc.stdout.splitlines() if 'v_cndmask_b32_sdwa' in line]
    check('llvm-mc prints each select as written', printed == [assembly(s) for s in SELECTS], '\n'.join(printed))
    for mode in ('reference', *DESTINATION_MODES):
        program = destination_program(mode)
        source = '\n'.join(line for line, _ in program) + '\n'
        check(f'destination ({mode}) words match llvm-mc', assemble(llvm_mc, source) == b''.join(raw for _, raw in program))


def run_pixel_words(renderer, adapter, vs, ps, expected_path):
    run = subprocess.run(
        [renderer, adapter, '--startup-pixel-words', str(vs), str(ps), str(WIDTH), str(HEIGHT), str(expected_path)],
        capture_output=True,
        text=True,
    )
    if run.returncode == 77:
        print(run.stdout + run.stderr, end='', flush=True)
        sys.exit(77)
    if 'debug_layer=enabled' in run.stdout:
        assert 'debug_errors=0 debug_warnings=0' in run.stdout, (run.stdout, run.stderr)
    return run


def draw_fixture(label, shader, model, modes, renderer, adapter, vs, kind, compiler, profile, directory):
    def expected_file(mode):
        path = directory / f'{label}_{mode}.bin'
        path.write_bytes(struct.pack(f'<{WIDTH * HEIGHT * 4}I', *model(mode)))
        return path

    ref_bin = expected_file('reference')
    prefix = f'{kind} ({adapter}) {label}'
    for idx, mode in enumerate(('reference', *modes)):
        ps = compile_hlsl(compiler, kind, shader(mode), f'{label}_{idx}_{kind}', profile, directory)
        run = run_pixel_words(renderer, adapter, vs, ps, ref_bin)
        if mode == 'reference':
            check(f'{prefix}: 16x16 pixel draw matches all 1,024 model words', run.returncode == 0, run.stdout + run.stderr)
            print(run.stdout, end='', flush=True)
            continue
        check(f'{prefix} mutation ({mode}) fails against the model',
              run.returncode == 1 and 'pixel mismatch' in run.stderr, run.stdout + run.stderr)
        own_run = run_pixel_words(renderer, adapter, vs, ps, expected_file(mode))
        check(f'{prefix} mutation ({mode}) matches its own model', own_run.returncode == 0, own_run.stdout + own_run.stderr)


def check_draw(dxc, renderer, adapter, fxc=None):
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    out_dir = HERE.parent / 'out'
    out_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sdwa-cndmask-', dir=out_dir) as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            vs = compile_hlsl(compiler, kind, FULLSCREEN_VS, f'vs_{kind}', vs_profile, directory)
            for fixture in (('preserve', pixel_hlsl, expected_pixel_words, MODES),
                            ('destination', destination_hlsl, destination_pixel_words, DESTINATION_MODES)):
                draw_fixture(*fixture, renderer, adapter, vs, kind, compiler, ps_profile, directory)


def main():
    if len(sys.argv) == 1:
        controls()
    elif len(sys.argv) == 3 and sys.argv[1] == 'binary':
        check_binary(sys.argv[2])
    elif len(sys.argv) in (5, 6) and sys.argv[1] == 'draw':
        controls()
        check_draw(*sys.argv[2:])
    else:
        raise SystemExit(__doc__)

    if failures:
        print('FAIL')
        for text in failures:
            print('  ' + text)
        sys.exit(1)
    print('PASS')


if __name__ == '__main__':
    main()
