"""Draw mixed-axis comparison borders from #2549. Arguments: renderer backend dxc fxc llvm-mc."""
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import startup_image_sample_c_border_check as fixture


DEPTHS = ((0.125, 0.25), (0.375, 0.5))
PROBES = ((0.5, 0.5), (0.0, 0.25), (0.25, 0.0), (0.0, 0.0))


def sample(u, v, border_x, border_value, reference):
    x, y = 2 * u - 0.5, 2 * v - 0.5
    i, j = math.floor(x), math.floor(y)
    total = 0.0
    for yi in (j, j + 1):
        for xi in (i, i + 1):
            outside = (xi < 0 or xi > 1) if border_x else (yi < 0 or yi > 1)
            depth = border_value if outside else DEPTHS[min(max(yi, 0), 1)][min(max(xi, 0), 1)]
            if reference < depth:
                total += (1 - abs(x - xi)) * (1 - abs(y - yi))
    return total


def expected(border_x, border_value, reference):
    return fixture.f32_hex(*(sample(x, y, border_x, border_value, reference) for x, y in PROBES))


def verify(renderer, backend, dxc, fxc, llvm_mc):
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        for compiler, kind, profile in ((dxc, 'dxc', '6_0'), (fxc, 'fxc', '5_1')):
            vertex = fixture.compile_hlsl(compiler, kind, fixture.FULLSCREEN_VS, 'vs_' + kind,
                                          'vs_' + profile, folder)
            cases = {}
            for border_x in (False, True):
                x, y = (fixture.CLAMP_BORDER, fixture.CLAMP_LAST_TEXEL) if border_x else (
                    fixture.CLAMP_LAST_TEXEL, fixture.CLAMP_BORDER)
                for color, word3 in ((0, fixture.OPAQUE_BLACK), (1, fixture.OPAQUE_WHITE)):
                    assembly, words = fixture.build_bilinear_program(x | (y << 3) | fixture.LESS, w3=word3)
                    assert fixture.assemble(llvm_mc, assembly) == words
                    program = fixture.G.Pipeline(fixture.VS_WORDS, words, fixture.root_blob())
                    pixel = fixture.compile_hlsl(compiler, kind, program.hlsl_ps(), f'ps_{kind}_{x}_{color}',
                                                 'ps_' + profile, folder)
                    cases[border_x, color] = pixel, fixture.manifest_sampler(program)
            good = swapped = 0
            for (border_x, color), (pixel, sampler) in cases.items():
                for reference in (0.0625, 0.3125, 0.75):
                    wanted = expected(border_x, color, reference)
                    args = [renderer, backend, '--startup-image-2d', vertex, pixel, str(reference), wanted]
                    result = subprocess.run([*args, sampler], capture_output=True, text=True)
                    if result.returncode == 77:
                        raise SystemExit(77)
                    assert result.returncode == 0, (args, sampler, result.stdout, result.stderr)
                    good += 1
                    wrong = expected(not border_x, color, reference)
                    if wrong == wanted:
                        continue
                    result = subprocess.run([*args, cases[not border_x, color][1]], capture_output=True, text=True)
                    assert result.returncode not in (0, 77) and 'actual=' + wrong in result.stderr, result.stderr
                    swapped += 1
            print(f'{kind} {backend}: {good} exact four-channel readbacks; {swapped} axis swaps caught', flush=True)


if __name__ == '__main__':
    verify(*sys.argv[1:])
