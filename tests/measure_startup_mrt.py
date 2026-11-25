"""Time the child processes of the existing startup MRT fixture on a Windows host."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile
import time
from unittest.mock import patch

import startup_mrt_check as mrt


def measure(consumer, backend, kind, dxc, fxc, report):
    run_child = subprocess.run
    counts = Counter()
    seconds = Counter()
    debug = set()

    def timed(args, **kwargs):
        name = Path(args[0]).name.lower()
        group = "gfx2hlsl --pending" if "--pending" in args else name
        start = time.perf_counter()
        child = run_child(args, **kwargs)
        seconds[group] += time.perf_counter() - start
        counts[group] += 1
        if name == Path(consumer).name.lower():
            debug.update(line.strip() for line in child.stderr.splitlines() if "debug layer:" in line)
        return child

    start = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as work, patch.object(subprocess, "run", timed):
        rc = mrt.check(Path(work), consumer, backend, kind, dxc, fxc)
    elapsed = time.perf_counter() - start
    if rc:
        raise SystemExit(rc)
    expected = {Path(consumer).name.lower(), Path(fxc if kind == "fxc" else dxc).name.lower(), "gfx2hlsl --pending"}
    assert counts.keys() == expected and debug, (counts, debug)
    result = dict(backend=backend, compiler=kind, seconds=round(elapsed, 4),
                  other_seconds=round(elapsed - sum(seconds.values()), 4), debug=sorted(debug),
                  children={k: {"count": counts[k], "seconds": round(seconds[k], 4)} for k in sorted(counts)})
    report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("MRT_TIMING", json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("consumer")
    parser.add_argument("backend", choices=("warp", "hardware"))
    parser.add_argument("kind", choices=("fxc", "dxc"))
    parser.add_argument("dxc")
    parser.add_argument("fxc")
    parser.add_argument("report", type=Path)
    measure(**vars(parser.parse_args()))
