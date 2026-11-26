"""tests/gpu_case_runner.py: the served cases answer like one-shot calls, and every failure of the server falls back to one.

The renderer here is a fake offscreen_check (a Python script behind a one-line launcher) that speaks the `--serve` protocol of
tests/offscreen_check.cpp: a banner closed by the marker pair of case -1, then per case the case text, "X2WEND <i>" on stderr and
"X2WCASE <i> <exit code>" on stdout. Its one-shot mode prints the same text, so the two paths can be compared.
"""
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_case_runner as R  # noqa: E402

FAKE = r'''
import os, sys
mode = os.environ.get("FAKE_MODE", "")
adapter, args = sys.argv[1], sys.argv[2:]

def one(args):
    """The text and exit code of one case, as the real tool prints them."""
    op = args[0]
    if op == "--pass":
        return "ok %s\n" % args[1], "", 0
    if op == "--mismatch":
        return "", "startup_flat_param mismatch %s\n" % args[1], 1
    if op == "--skip":
        return "skip: nothing\n", "", 77
    return "", "Unknown GPU operation or invalid arguments: %s\n" % op, 2

banner = "backend=%s debug_layer=unavailable\nadapter=Fake\n" % adapter
if args and args[0] == "--serve":
    if mode == "no-serve":
        sys.stderr.write("Unknown GPU operation or invalid arguments: --serve\n")
        sys.exit(2)
    if mode == "no-adapter":
        print("No D3D12-capable hardware adapter visible")
        sys.exit(77)
    sys.stdout.write(banner)
    sys.stderr.write("XBOX2WINDOWS_ADAPTER: skipping Other\n")
    sys.stderr.write("X2WEND -1\n")
    print("X2WCASE -1 0", flush=True)
    for index, line in enumerate(sys.stdin):
        out, err, code = one(line.rstrip("\n").split("\t"))
        if mode == "die-at-%d" % index:
            os._exit(3)
        sys.stdout.write(out)
        sys.stderr.write(err)
        sys.stderr.write("X2WEND %d\n" % index)
        print("X2WCASE %d %d" % (index, code), flush=True)
    sys.exit(0)
out, err, code = one(args)
sys.stdout.write("ONESHOT " + banner + out)
sys.stderr.write(err)
sys.exit(code)
'''


def launcher(directory, name, mode):
    fake = directory / 'fake_offscreen.py'
    fake.write_text(FAKE)
    if os.name == 'nt':
        path = directory / (name + '.cmd')
        path.write_text('@echo off\r\nset FAKE_MODE=%s\r\n"%s" "%s" %%*\r\n' % (mode, sys.executable, fake))
    else:
        path = directory / name
        path.write_text('#!/bin/sh\nFAKE_MODE=%s exec "%s" "%s" "$@"\n' % (mode, sys.executable, fake))
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class RunnerChecks(unittest.TestCase):
    def setUp(self):
        R._servers.clear()
        self.addCleanup(R._servers.clear)
        self.addCleanup(R._close_all)          # runs first: the servers of the test are closed before the table is emptied
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def run_case(self, renderer, *args):
        return R.run(renderer, 'warp', args)

    def test_served_cases_carry_their_own_text_and_exit_code(self):
        renderer = launcher(self.dir, 'fake', 'serve')
        first = self.run_case(renderer, '--pass', 'a')
        second = self.run_case(renderer, '--mismatch', 'b')
        third = self.run_case(renderer, '--pass', 'c')
        self.assertEqual((first.returncode, second.returncode, third.returncode), (0, 1, 0))
        self.assertEqual(first.stdout, 'backend=warp debug_layer=unavailable\nadapter=Fake\nok a\n')   # banner once
        self.assertEqual(first.stderr, 'XBOX2WINDOWS_ADAPTER: skipping Other\n')
        self.assertEqual((second.stdout, second.stderr), ('', 'startup_flat_param mismatch b\n'))
        self.assertEqual((third.stdout, third.stderr), ('ok c\n', ''))
        self.assertEqual(self.run_case(renderer, '--bogus').returncode, 2)

    def test_skip_and_missing_adapter_keep_their_exit_code(self):
        self.assertEqual(self.run_case(launcher(self.dir, 'skip', 'serve'), '--skip').returncode, 77)
        none = self.run_case(launcher(self.dir, 'none', 'no-adapter'), '--pass', 'x')
        self.assertEqual(none.returncode, 77)
        self.assertIn('No D3D12-capable hardware adapter visible', none.stdout)

    def test_a_renderer_without_serve_falls_back_to_one_shot_calls(self):
        renderer = launcher(self.dir, 'old', 'no-serve')
        result = self.run_case(renderer, '--mismatch', 'z')
        self.assertEqual(result.returncode, 1)
        self.assertTrue(result.stdout.startswith('ONESHOT '))
        self.assertEqual(self.run_case(renderer, '--pass', 'q').returncode, 0)

    def test_a_server_that_dies_is_restarted_and_the_case_still_answers(self):
        renderer = launcher(self.dir, 'dies', 'die-at-1')
        self.assertEqual(self.run_case(renderer, '--pass', 'a').returncode, 0)
        again = self.run_case(renderer, '--mismatch', 'b')            # case 1 kills the first server; the restart answers it
        self.assertEqual((again.returncode, again.stderr.splitlines()[-1]), (1, 'startup_flat_param mismatch b'))

    def test_every_exit_leaves_no_process_and_no_open_pipe(self):
        made = []
        real = R._Server

        def record(*args):
            made.append(real(*args))
            return made[-1]

        with unittest.mock.patch.object(R, '_Server', record):
            self.run_case(launcher(self.dir, 'old', 'no-serve'), '--pass', 'a')      # failed handshake, then the one-shot call
            self.run_case(launcher(self.dir, 'none', 'no-adapter'), '--pass', 'a')   # skip 77 at the handshake
            dies = launcher(self.dir, 'dies', 'die-at-1')
            self.run_case(dies, '--pass', 'a')
            self.run_case(dies, '--mismatch', 'b')                                   # dead server, restart
            R._close_all()                                                           # the restarted server, at exit
        self.assertEqual(len(made), 4)
        for server in made:
            self.assertIsNotNone(server.proc.poll())
            self.assertTrue(all(s.closed for s in (server.proc.stdin, server.proc.stdout, server.proc.stderr)))

    def test_x2w_flat_server_zero_uses_one_shot_calls(self):
        renderer = launcher(self.dir, 'off', 'serve')
        R.SERVER_OFF = True
        self.addCleanup(setattr, R, 'SERVER_OFF', False)
        self.assertTrue(self.run_case(renderer, '--pass', 'a').stdout.startswith('ONESHOT '))


if __name__ == '__main__':
    unittest.main()
