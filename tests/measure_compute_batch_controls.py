"""Exercise the compute batch with a case-local debug error and logical device removal."""
from pathlib import Path
import re
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import compute_load_format_check as fixture


def check(compiler, renderer, backend, directory):
    shader = directory / 'load.cso'
    source = directory / 'load.hlsl'
    source.write_text(fixture.lower(), encoding='ascii')
    subprocess.run([compiler, '-T', 'cs_6_0', '-E', 'main', '-Fo', shader, source], check=True)
    cases = []
    for index, groups in enumerate((1, 2, 1)):
        path = directory / f'{index}.case'
        path.write_bytes(struct.pack('<657I', groups, *fixture.expected_words(groups),
                                     *fixture.constants(), *fixture.source_words()))
        cases.append(path)

    for mode, wanted in (('--compute-store-batch', ['pass', 'pass', 'pass']),
                         ('--compute-batch-debug-control', ['pass', 'error', 'pass']),
                         ('--compute-batch-removal-control', ['pass', 'error', 'error'])):
        run = subprocess.run([renderer, backend, mode, shader, *cases], capture_output=True, text=True, timeout=40)
        if mode == '--compute-batch-debug-control' and run.returncode == 77:
            assert 'requires the D3D12 debug layer' in run.stdout and 'debug_layer=unavailable' in run.stdout
            print(mode, 'skip (77), debug layer unavailable')
            continue
        statuses = re.findall(r'^case=(\d+) status=(\w+)$', run.stdout, re.MULTILINE)
        assert run.returncode == int('error' in wanted) and statuses == list(zip(('0','1','2'), wanted)), run
        errors = re.split(r'^case=\d+ path=.*\n', run.stderr, flags=re.MULTILINE)
        assert len(errors) == 4
        if mode == '--compute-batch-debug-control':
            assert [s.count('pixel batch control error') for s in errors] == [0, 0, 1, 0], errors
            assert re.findall(r'^debug_errors=(\d+) debug_warnings=\d+\ncase=\d+ status=', run.stdout,
                              re.MULTILINE) == ['0', '1', '0']
        if mode == '--compute-batch-removal-control':
            assert 'Compute batch device failed (HRESULT 0x887a0005)' in errors[3], errors
        print(mode, wanted, run.returncode)


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        check(*sys.argv[1:], Path(tmp))
