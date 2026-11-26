"""Compare each actual renderer call, a one-bit mutation and the following valid draw."""
import argparse
import contextlib
import os
from pathlib import Path
import subprocess
import tempfile
import time
import gpu_case_runner as runner
import startup_gfx2hlsl_f16_alu_draw_check as half
import startup_gfx2hlsl_f32_nan_check as nan
import startup_gfx2hlsl_f32_sdwa_check as sdwa
import startup_image_sample_b_check as bias

parser = argparse.ArgumentParser(description=__doc__)
for tool in ('renderer', 'dxc', 'fxc'):
    parser.add_argument(tool, type=Path)
parser.add_argument('device', help='WARP or a hardware adapter name substring')
parser.add_argument('output', type=Path)
params = parser.parse_args()
renderer, dxc, fxc = (p.resolve() for p in (params.renderer, params.dxc, params.fxc))
out = params.output
out.mkdir(parents=True, exist_ok=True)
adapter = 'warp' if params.device == 'WARP' else 'hardware'
os.environ['XBOX2WINDOWS_ADAPTER'] = params.device
served = runner.run

def compare(args):
    def once():
        return subprocess.run([str(renderer), adapter, *map(str,args)], capture_output=True, text=True)
    calls = (('one', once), ('served', lambda: served(renderer,adapter,args)))
    if stats['pairs'] % 2:
        calls = reversed(calls)
    results = {}
    for name, func in calls:
        start = time.perf_counter()
        result = func()
        stats[name] += time.perf_counter()-start
        results[name] = (result.returncode, result.stdout, result.stderr)
    if results['one'] != results['served']:
        raise ValueError((args,results))
    server = runner._servers.get((str(renderer), adapter))
    assert server and server.ready and server.proc.poll() is None, 'runner used fallback'
    stats['pairs'] += 1
    return subprocess.CompletedProcess(args, *results['served'])

def paired(_renderer, _adapter, args):
    good = compare(args)
    stats['original'] += 1
    if good.returncode or stats['original'] % 7 != 1:
        return good
    altered = list(args)
    index = {'--startup-flat-param': 3, '--startup-pixel-words': 5, '--startup-image-2d-array-mips': 4}[args[0]]
    if args[0] == '--startup-pixel-words':
        content = bytearray(Path(args[index]).read_bytes())
        content[0] ^= 1
        altered[index] = Path(tmp)/'mutant.bin'
        altered[index].write_bytes(content)
    else:
        words = str(args[index]).split(',')
        words[0] = f'{int(words[0],16)^1:08x}'
        altered[index] = ','.join(words)
    assert compare(altered).returncode == 1, 'wrong-value control did not fail'
    assert compare(args).returncode == 0, 'valid case failed after wrong-value control'
    stats['restored'] += 1
    return good

runner.run = paired
with tempfile.TemporaryDirectory() as tmp:
    for name, suite in [('bias',bias),('half',half),('nan',nan),('sdwa',sdwa)]:
        stats = dict(original=0, pairs=0, restored=0, served=0.0, one=0.0)
        with (out/f'{params.device}-{name}.log').open('w') as log, contextlib.redirect_stdout(log):
            suite.check_draw(dxc, renderer, adapter, fxc)
        print(name, stats, flush=True)
        runner._close_all()
        runner._servers.clear()
