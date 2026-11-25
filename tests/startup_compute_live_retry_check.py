"""Dispatch twice through the same device slot 67 pipeline with an offline translation arriving between calls."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time

from startup_compute_dispatch_check import C, G, prepare_case


def words(output):
    return list(struct.unpack(f'<{C.WORDS}I', output.read_bytes()))


def run(directory, consumer, dxc, adapter, publish):
    cache, env = prepare_case(directory, adapter)
    output = directory / 'out.bin'
    args = [consumer, directory / 'vertex', directory / 'pixel', directory / 'root', 'slot67',
            directory / 'program', directory / 'data', directory / 'state', 'dispatch-retry']
    stdout, stderr = directory / 'stdout', directory / 'stderr'
    sentinel = [C.SENTINEL] * C.WORDS
    with stdout.open('wb') as out, stderr.open('wb') as err, subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=out, stderr=err, env=dict(env, X2W_TEST_DISPATCH_OUT=str(output))) as process:
        try:
            deadline = time.monotonic() + 20
            while 'first slot 67 Dispatch fenced' not in stdout.read_text():
                if process.poll() == 77:
                    print(stdout.read_text(), end='')
                    raise SystemExit(77)
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError(f'first Dispatch did not fence\n{stderr.read_text()[-2500:]}')
                time.sleep(0.025)
            first = time.monotonic()
            assert words(output) == sentinel, 'first Dispatch wrote before translation'
            output.unlink()
            request, = cache.glob('req_*.c67')
            key = request.stem.removeprefix('req_')
            assert not (cache / (key + '.txt')).exists()
            if publish:
                result = subprocess.run([sys.executable, '-B', G.__file__, '--pending', cache], check=True,
                                        capture_output=True, text=True, env=dict(os.environ, X2W_DXC=dxc))
                assert f'{key}: ok' in result.stdout, result.stdout
            # The runtime polls a missing manifest at one-second intervals.
            time.sleep(max(0, first + 1.1 - time.monotonic()))
            process.communicate(input=b'\n', timeout=20)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    log = stderr.read_text()
    assert process.returncode == 0, log[-2500:]
    assert log.count('Dispatch(3, 2, 2) refused: no translation') == (1 if publish else 2), log[-2500:]
    assert log.count('Dispatch(3, 2, 2) translated:') == int(publish), log[-2500:]
    assert words(output) == (C.expected('A') if publish else sentinel)
    errors = [line for line in log.splitlines() if 'x2w-d3d12-debug ' in line and (' sev=0 ' in line or ' sev=1 ' in line)]
    assert not errors, errors
    print(stdout.read_text(), end='')
    print(f'{key}: both fenced {C.WORDS}-word readbacks match, late publication={publish}')


if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[3] not in ('warp', 'hardware'):
        sys.exit('usage: startup_compute_live_retry_check.py <consumer> <dxc> <warp|hardware>')
    for publish in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            run(Path(tmp), *sys.argv[1:], publish)
