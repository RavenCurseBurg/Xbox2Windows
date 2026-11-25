"""Record the actual compiler inputs, ordered GPU inputs and outcomes in an unmodified checkout."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

root, output, inputs, compiler, renderer = map(Path, sys.argv[1:6])
adapter = sys.argv[6]
inputs.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(root.resolve() / "tests"))
import pixel_aot_check as pixel

draws = []
compiles = []
launches = []
run = subprocess.run

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def capture(cmd, *args, **kwargs):
    folder = Path(kwargs.get("cwd", Path.cwd()))
    result = run(cmd, *args, **kwargs)
    if Path(cmd[0]).name.lower() == compiler.name.lower():
        assert result.returncode == 0
        compiles.append(digest(folder / cmd[-1]))
        return result
    assert Path(cmd[0]).name.lower() == renderer.name.lower(), cmd
    if cmd[2] == "--check-pixel-batch":
        pairs = [(cmd[3], case) for case in cmd[4:]]
    else:
        assert cmd[2] == "--check-pixel-pairs" and (len(cmd) - 3) % 2 == 0
        pairs = list(zip(cmd[3::2], cmd[4::2]))
    rows = re.findall(r"^case=(\d+) status=(\w+)$", result.stdout, re.M)
    assert [int(i) for i, _ in rows] == list(range(len(pairs))), result.stdout
    for (shader, case), (_, status) in zip(pairs, rows):
        draws.append((digest(folder / shader), digest(folder / case), status))
    launches.append(cmd[2])
    return result

start = perf_counter()
with patch.object(subprocess, "run", capture):
    pixel.check_int_predicate_draws(inputs.resolve(), str(compiler.resolve()), str(renderer.resolve()), adapter)
duration = perf_counter() - start
assert len(compiles) == 8 and len(draws) == 33
assert [r[2] for r in draws].count("pass") == 32
assert [r[2] for r in draws].count("mismatch") == 1
output.with_suffix(".json").write_text(json.dumps(dict(compiles=compiles, draws=draws,
                                                      forms=sorted(pixel.ACCEPTED)), indent=2))
print(f"{output.stem}: {len(launches)} renderer launches in {duration:.3f}s, "
      f"8 compiles, 32 pass, 1 mismatch, {len(pixel.ACCEPTED)} recorded forms")
