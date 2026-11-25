"""v_cube{id,sc,tc,ma}_f32 in tools/startup/gfx2hlsl.py.

  (no arguments)                       translation controls, pure Python
  binary <llvm-mc>                     reassemble the fixture sources and compare them with the words below
  draw <dxc> <offscreen_check> <warp|hardware> [fxc]
                                       compile the translated pair and compare four output words per
                                       direction on a source-built full-screen triangle

The four operations follow LLVM's ConstantFoldAMDGCNCubeIntrinsic. The major axis is z when |z| is at
least |x| and |y|, otherwise y when |y| is at least |x|, otherwise x. A face is negative only for a
nonzero, non-NaN negative coordinate. cubema returns twice the signed major coordinate.
"""
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
sys.path.insert(0, str(HERE))
import gfx2hlsl as G  # noqa: E402
from startup_flat_param_check import assemble, compile_hlsl, draw, root_blob  # noqa: E402
from startup_gfx2hlsl_bit_alu_check import (  # noqa: E402
    POST_SOURCE, POST_WORDS, PRE_SOURCE, PRE_WORDS, vs_source, vs_words,
)

OPS = ('v_cubeid_f32', 'v_cubesc_f32', 'v_cubetc_f32', 'v_cubema_f32')
BODY_SOURCE = '''\
v_cubeid_f32 v12, v0, v4, v8
v_cubesc_f32 v13, v1, v5, v9
v_cubetc_f32 v14, v2, v6, v10
v_cubema_f32 v15, v3, v7, v11
'''
BODY_WORDS = bytes.fromhex(
    '0c0044d500092204' '0d0045d5010b2604' '0e0046d5020d2a04' '0f0047d5030f2e04')
# The first pair is swapped as a negative control: the same inputs then write sc to v12 and id to v13.
BAD_BODY = bytes.fromhex(
    '0c0045d500092204' '0d0044d5010b2604' '0e0046d5020d2a04' '0f0047d5030f2e04')

# Each direction covers a face; the first group also contains y/x and z/x ties and -0.0 on the major axis.
DIRECTIONS = (
    (2.0, -2.0, 1.0),
    (-1.5, 0.25, -1.5),
    (-3.0, 1.0, 2.0),
    (0.0, 0.0, -0.0),
    (1.0, 0.5, -0.25),
    (0.25, 1.0, 0.5),
    (0.5, -1.0, -0.5),
    (-0.25, -0.5, 1.0),
)


class Probe(G.Stage):
    """A Stage with no pipeline: enough state for one instruction to be translated."""

    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True


def one(assembly):
    Probe().one(assembly.split(' ', 1)[0], assembly)


def refused(assembly):
    try:
        one(assembly)
    except G.Unsupported as reason:
        return str(reason)
    return None


def word(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def direction_words(directions):
    """The flat parameters are 32-bit words, so pass the binary32 bits of each coordinate."""
    return tuple((word(x), word(y), word(z)) for x, y, z in directions)


def cube_reference(x, y, z):
    if abs(z) >= abs(x) and abs(z) >= abs(y):
        face, sc, tc, ma = (5.0, -x, -y, z) if z < 0 else (4.0, x, -y, z)
    elif abs(y) >= abs(x):
        face, sc, tc, ma = (3.0, x, -z, y) if y < 0 else (2.0, x, z, y)
    else:
        face, sc, tc, ma = (1.0, z, -y, x) if x < 0 else (0.0, -z, -y, x)
    return [face, sc, tc, ma + ma]


def expected_text(directions):
    return ','.join('%08x' % word(value) for d in directions for value in cube_reference(*d))


def translate(directions, body=BODY_WORDS):
    return G.Pipeline(vs_words(direction_words(directions)), PRE_WORDS + body + POST_WORDS, root_blob())


def controls():
    checks = {
        'v_cubeid_f32': ('? 5.0f : 4.0f', '? 3.0f : 2.0f', '? 1.0f : 0.0f'),
        'v_cubesc_f32': ('-cube0.x', 'cube0.x', '-cube0.z', 'cube0.z'),
        'v_cubetc_f32': ('-cube0.y', '-cube0.z', 'cube0.z'),
        'v_cubema_f32': ('* 2.0f',),
    }
    for op in OPS:
        stage = Probe()
        stage.one(op, '%s v12, v0, v4, v8' % op)
        assert stage.lines[0] == 'float3 cube0 = float3(F(v0), F(v4), F(v8));', stage.lines
        assert stage.lines[1].startswith('v12 = U('), stage.lines
        for needle in checks[op]:
            assert needle in stage.lines[1], (op, needle, stage.lines)
        print('%s: translated' % op)

    clamp = Probe()
    clamp.one('v_cubeid_f32', 'v_cubeid_f32 v12, v0, v4, v8 clamp')
    assert 'saturate(' in clamp.lines[1], clamp.lines
    print('clamp: saturates the result')

    modified = Probe()
    modified.one('v_cubeid_f32', 'v_cubeid_f32 v12, -v0, |v4|, v8')
    assert '(-F(v0))' in modified.lines[0] and 'abs(F(v4))' in modified.lines[0], modified.lines
    print('source neg/abs: applied before the cube operation')

    separate = Probe()
    separate.one('v_cubeid_f32', 'v_cubeid_f32 v12, v0, v4, v8')
    first_temp = separate.lines[0]
    separate.lines = []
    separate.one('v_cubema_f32', 'v_cubema_f32 v13, v1, v5, v9')
    assert 'float3 cube0' in first_temp and 'float3 cube1' in separate.lines[0], (first_temp, separate.lines)
    print('separate blocks: temporary names stay unique')

    for assembly, needle in (
        ('v_cubeid_f32 v12, v0, v4', 'expected 4'),
        ('v_cubeid_f32 v256, v0, v4, v8', 'destination'),
        ('v_cubeid_f32 v12, v0, v4, v8 mul:2', 'with mul:2'),
        ('v_cubeid_f32 v12, v0, v4, v8 op_sel:[1,0,0]', 'with op_sel'),
    ):
        reason = refused(assembly)
        assert reason and needle in reason, (assembly, reason)
        print('refused: %s (%s)' % (assembly, reason))


def check_binary(llvm_mc):
    assert assemble(llvm_mc, BODY_SOURCE) == BODY_WORDS
    assert assemble(llvm_mc, PRE_SOURCE) == PRE_WORDS
    assert assemble(llvm_mc, POST_SOURCE) == POST_WORDS
    for index in range(0, len(DIRECTIONS), 4):
        directions = DIRECTIONS[index:index + 4]
        assert assemble(llvm_mc, vs_source(direction_words(directions))) == vs_words(direction_words(directions))
    print('fixture words match llvm-mc')


def check_draw(dxc, renderer, adapter, fxc=None):
    controls()
    compilers = ([('fxc', fxc, 'vs_5_1', 'ps_5_1')] if fxc else []) + [('dxc', dxc, 'vs_6_0', 'ps_6_0')]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for kind, compiler, vs_profile, ps_profile in compilers:
            for index, direction in enumerate(DIRECTIONS):
                pipe = translate((direction,) * 4)
                vs = compile_hlsl(compiler, kind, pipe.hlsl_vs(),
                                  'cube_vs_%d_%s' % (index, kind), vs_profile, directory)
                ps = compile_hlsl(compiler, kind, pipe.hlsl_ps(),
                                  'cube_ps_%d_%s' % (index, kind), ps_profile, directory)
                expected = expected_text((direction,))
                run = draw(renderer, adapter, vs, ps, expected)
                if run.returncode:
                    raise AssertionError('%s cube direction %s\n%s\n%s' % (kind, direction, run.stdout, run.stderr))
                print('%s cube direction %s: %s' % (kind, direction, expected), flush=True)

            first = DIRECTIONS[0]
            first_pipe = translate((first,) * 4)
            first_vs = compile_hlsl(compiler, kind, first_pipe.hlsl_vs(),
                                    'cube_first_vs_%s' % kind, vs_profile, directory)
            first_expected = expected_text((first,))

            bad_ps = compile_hlsl(compiler, kind, translate((first,) * 4, BAD_BODY).hlsl_ps(),
                                  'cube_bad_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, first_vs, bad_ps, first_expected)
            if run.returncode == 0:
                raise AssertionError('%s swapped id/sc did not mismatch\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s swapped id/sc: mismatch as expected' % kind, flush=True)

            tie = DIRECTIONS[1]
            tie_pipe = translate((tie,) * 4)
            tie_vs = compile_hlsl(compiler, kind, tie_pipe.hlsl_vs(),
                                  'cube_tie_vs_%s' % kind, vs_profile, directory)
            tie_ps = compile_hlsl(compiler, kind, tie_pipe.hlsl_ps().replace('>=', '>'),
                                  'cube_tie_ps_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, tie_vs, tie_ps, expected_text((tie,)))
            if run.returncode == 0:
                raise AssertionError('%s tie-order control did not mismatch\n%s\n%s' % (kind, run.stdout, run.stderr))
            print('%s tie-order control: mismatch as expected' % kind, flush=True)

            doubled_ps = compile_hlsl(compiler, kind, first_pipe.hlsl_ps().replace('* 2.0f', '* 1.0f'),
                                      'cube_double_%s' % kind, ps_profile, directory)
            run = draw(renderer, adapter, first_vs, doubled_ps, first_expected)
            if run.returncode == 0:
                raise AssertionError('%s cubema factor control did not mismatch\n%s\n%s'
                                     % (kind, run.stdout, run.stderr))
            print('%s cubema factor control: mismatch as expected' % kind, flush=True)


def main():
    if len(sys.argv) == 1:
        controls()
        return
    mode, *args = sys.argv[1:]
    if mode == 'binary':
        check_binary(*args)
    elif mode == 'draw':
        check_draw(*args)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
