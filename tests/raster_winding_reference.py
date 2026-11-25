"""Generate the cull/winding reference for issue 1999 using explicit desktop PSO state."""
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory


VERTEX = """
float4 main(uint id : SV_VertexID) : SV_Position {
    float2 points[3] = {float2(-1, 1), float2(-0.0625, 1), float2(-1, -1)};
    float2 p = points[id % 3];
    if (id >= 3) p.x = -p.x;
    return float4(p, 0.5, 1);
}
"""
PIXEL = """
float4 main(bool front : SV_IsFrontFace) : SV_Target {
    return front ? float4(0, 1, 0, 1) : float4(1, 0, 0, 1);
}
"""
# The left triangle's screen vertices are (0,0), (3.75,0) and (0,4).
# The right triangle is its mirror across x=4. No pixel center lies on a diagonal.
COVERAGE = ("LLL..RRR", "LL....RR", "L......R", "........")
COLORS = {".": (-1.,) * 4, "F": (0., 1., 0., 1.), "B": (1., 0., 0., 1.)}
EXPECTED = {
    ("none", "cw"): "FB", ("front", "cw"): ".B", ("back", "cw"): "F.",
    ("none", "ccw"): "BF", ("front", "ccw"): "B.", ("back", "ccw"): ".F",
}


def reference(pair):
    labels = {".": ".", "L": pair[0], "R": pair[1]}
    return b"".join(struct.pack("<4f", *COLORS[labels[pixel]]) for row in COVERAGE for pixel in row)


def verify(dxc, renderer, adapter, directory):
    compiled = []
    for name, text in (("vs", VERTEX), ("ps", PIXEL)):
        source, shader = directory / f"{name}.hlsl", directory / f"{name}.cso"
        source.write_text(text)
        subprocess.run([dxc, "-E", "main", "-T", f"{name}_6_0", "-Fo", shader, source], check=True)
        compiled.append(shader)

    fixture = directory / "expected.bin"
    command = [renderer, adapter, "--raster-reference", *compiled, "8", "4", fixture]
    def draw(state, pair, mismatch):
        fixture.write_bytes(reference(pair))
        result = subprocess.run([*command, *state], capture_output=True, text=True, timeout=20)
        if adapter == "hardware" and result.returncode == 77:
            print(result.stdout + result.stderr)
            raise SystemExit(77)
        report = result.stdout + result.stderr
        required = "pixel mismatch (" if mismatch else "all 32 pixels matched"
        if result.returncode != int(mismatch) or required not in report:
            raise AssertionError(report)

    for state, pair in EXPECTED.items():
        draw(state, pair, False)
        print(f"{adapter} {state}: 32 pixels matched")
    for changed, expected in (
        (("none", "cw"), ("front", "cw")),
        (("none", "cw"), ("back", "cw")),
        (("back", "cw"), ("front", "cw")),
        (("back", "cw"), ("back", "ccw")),
    ):
        draw(changed, EXPECTED[expected], True)
        print(f"{adapter} {changed} rejected against {expected}")


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        verify(*sys.argv[1:], Path(directory))
