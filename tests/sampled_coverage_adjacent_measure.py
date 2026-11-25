"""Run the thirteen no-output cases from #737 on the instrumented renderer at d527420e."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import sampled_coverage_check as fixture


def inputs():
    for slot in range(2):
        source = fixture.lower(fixture.decoded(fixture.PROGRAM), fixture.bindings(slot))
        for scale, threshold in ((1, .5), (.5, .375), (1, 0), (0, .25), (-1, -.5)):
            yield 'main', slot, scale, threshold, source, fixture.case(slot, scale, threshold, 1, 6)
    for label, slot, base in (('red', 0, 0), ('red', 1, 0), ('red_base2', 0, 2)):
        source = fixture.lower(fixture.decoded(fixture.RED_PROGRAM), fixture.red_bindings(slot, scale_base=base))
        yield label, slot, 1, .75, source, fixture.case(slot, 1, .75, 1 + base, 10, red=True)


def measure(dxc, renderer, adapter, mode, repeat):
    env = dict(os.environ, X2W_COV_ADJACENT=str(int(mode == 'adjacent')))
    backend = 'warp' if adapter == 'warp' else 'hardware'
    if backend == 'hardware':
        env['XBOX2WINDOWS_ADAPTER'] = adapter
    else:
        env.pop('XBOX2WINDOWS_ADAPTER', None)
    failed = False
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        shaders = {}
        for label, slot, scale, threshold, source, reference in inputs():
            if source not in shaders:
                src = work / 'draw.hlsl'
                src.write_text(source)
                shader = work / f'draw{len(shaders)}.dxil'
                subprocess.run([dxc, '-T', 'ps_6_0', '-E', 'main', '-Fo', shader, src], check=True)
                shaders[source] = shader
            case = work / 'samples.bin'
            case.write_bytes(reference)
            for run in range(1, repeat + 1):
                p = subprocess.run([renderer, backend, '--sampled-coverage', shaders[source], case],
                                   env=env, capture_output=True, text=True)
                log = p.stdout + p.stderr
                rows = re.findall(r'coverage_record adjacent=(\d+) first=(\d+) second=(\d+) one_pixel_sum=(\d+)', log)
                if len(rows) != 1:
                    raise RuntimeError(f'expected one query record, got {len(rows)}\n{log}')
                adjacent, first, second, single_sum = map(int, rows[0])
                expected = sum(reference[-64:])
                if adjacent != int(mode == 'adjacent') or single_sum != expected:
                    raise RuntimeError(f'query mode or one-pixel sum differs\n{log}')
                if p.returncode != int(first != expected) or (p.returncode and 'Full draw coverage mismatch' not in log):
                    raise RuntimeError(f'unexpected fixture exit {p.returncode}\n{log}')
                if mode == 'base' and second != 0:
                    raise RuntimeError(f'empty baseline query returned {second}')
                failed |= bool(p.returncode) or (mode == 'adjacent' and second != expected)
                gpu = re.search(r'^adapter=(.*) vendor=(\w+) device=(\w+)$', log, re.M)
                driver = re.search(r'^driver=(.*)$', log, re.M)
                debug_layer = re.search(r'debug_layer=(\w+)', log)
                print(json.dumps(dict(mode=mode, run=run, adapter=gpu[1], vendor=gpu[2], device=gpu[3],
                                      driver=driver[1], debug=debug_layer[1], group=label, slot=slot, scale=scale,
                                      threshold=threshold, expected=expected, first=first, second=second,
                                      one_pixel_sum=single_sum, exit=p.returncode)), flush=True)
    return int(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dxc')
    parser.add_argument('renderer')
    parser.add_argument('adapter', help='hardware adapter name substring or warp')
    parser.add_argument('mode', choices=('base', 'adjacent'))
    parser.add_argument('--repeat', type=int, choices=range(1, 10), default=1)
    args = parser.parse_args()
    raise SystemExit(measure(args.dxc, args.renderer, args.adapter, args.mode, args.repeat))
