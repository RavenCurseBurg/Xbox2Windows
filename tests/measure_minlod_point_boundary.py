"""Locate a point-filter MinLod transition using the existing translated and direct pixel shaders.

Usage from the repository root:
    python -B tests/measure_minlod_point_boundary.py <spirv-as> <spirv-val> <spirv-cross> <dxc> <offscreen_check> <warp|hardware>

All inputs are exactly representable binary fractions. Each candidate mip must match all 64 pixels.
The fixture repeats its wrong-mip control for both shader paths at each grid resolution.
"""
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / 'tests' / 'spirv_window'))
import minlod_levels as fixture

render_args = sys.argv[1:]
measure_mip = fixture.point_mip
observed = {}

def record_mip(renderer, adapter, operation, dxil, clamp, levels, kind):
    level = measure_mip(renderer, adapter, operation, dxil, clamp, levels, kind)
    observed[kind, float(clamp)] = level
    return level

fixture.point_mip = record_mip

def probe(points):
    fixture.FRACTIONAL = tuple((format(p, '.12g'), (1, 2)) for p in points)
    status = fixture.main(*render_args, fractional=True)
    if status:
        raise SystemExit(status)
    assert all(observed['control', p] == observed['translated', p] for p in points)
    print('GRID', ' '.join(f"{p:.8f}:{observed['translated', p]}" for p in points), flush=True)

coarse = [i / 16 for i in range(16, 33)]
probe(coarse)
before = max(p for p in coarse if observed['translated', p] == 1)
after = min(p for p in coarse if observed['translated', p] == 2)
assert after - before == 1 / 16
fine = [before + i / 256 for i in range(17)]
probe(fine)
before = max(p for p in fine if observed['translated', p] == 1)
after = min(p for p in fine if observed['translated', p] == 2)
assert after - before == 1 / 256
print(f'BRACKET last mip 1={before:.8f}, first mip 2={after:.8f}, step=1/256', flush=True)
