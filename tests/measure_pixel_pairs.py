"""Replay the existing integer predicate draws in eight one-shader batches and one mixed batch."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import pixel_aot_check


def measure(dxc, renderer, adapter, directory):
    groups = []

    def save_group(_renderer, _adapter, shader, cases):
        saved = directory / f"group-{len(groups)}.dxil"
        shutil.copyfile(shader, saved)
        groups.append((saved, list(cases)))

    # The existing fixture overwrites its DXIL on each compile. Snapshot the bytes
    # before timing so the two paths draw exactly the same ordered input sequence.
    with patch.object(pixel_aot_check, "run_pixel_batch", save_group):
        pixel_aot_check.check_int_predicate_draws(directory, dxc, renderer, adapter)
    pairs = [(shader, case, expected) for shader, cases in groups for case, expected in cases]
    assert len(groups) == 8 and len(pairs) == 33 and sum(s == "mismatch" for _, _, s in pairs) == 1
    digest = hashlib.sha256()
    for shader, case, expected in pairs:
        for data in (shader.read_bytes(), case.read_bytes(), expected.encode("ascii")):
            digest.update(hashlib.sha256(data).digest())
    print(f"8 shaders, 33 draws, ordered input and expectation SHA-256 {digest.hexdigest()}", flush=True)
    singles = [("--check-pixel-batch", [shader, *(p for p, _ in cases)], [s for _, s in cases])
               for shader, cases in groups]
    mixed = [("--check-pixel-pairs", [v for shader, case, _ in pairs for v in (shader, case)],
              [s for _, _, s in pairs])]

    def replay(commands):
        start = perf_counter()
        for operation, paths, expected in commands:
            result = subprocess.run([renderer, adapter, operation, *(p.name for p in paths)],
                                    cwd=directory, capture_output=True, text=True)
            if result.returncode == 77 and adapter == "hardware":
                print(result.stdout + result.stderr)
                sys.exit(77)
            lines = [s for s in result.stdout.splitlines() if s.startswith("case=")]
            if (lines != [f"case={i} status={s}" for i, s in enumerate(expected)]
                    or result.returncode != int("mismatch" in expected)):
                raise AssertionError(result.stdout + result.stderr)
        return perf_counter() - start

    for iteration in range(4):
        order = [("eight", singles), ("one", mixed)]
        if iteration % 2:
            order.reverse()
        for name, commands in order:
            print(f"iteration={iteration} path={name} elapsed={replay(commands):.4f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dxc", type=Path)
    parser.add_argument("renderer", type=Path)
    parser.add_argument("adapter", choices=("hardware", "warp"))
    args = parser.parse_args()
    with TemporaryDirectory() as folder:
        measure(args.dxc.resolve(), args.renderer.resolve(), args.adapter, Path(folder))
