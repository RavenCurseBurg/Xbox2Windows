"""Verify packed binary16 conversion and channel order in the D3D12 pixel image-store fixture."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

import startup_pixel_image_store_check as fixture


VECTORS = (
    (0x0000, 0x8000, 0x0001, 0x8001),  # signed zero and least subnormals
    (0x03ff, 0x0400, 0x83ff, 0x8400),  # subnormal/normal boundary
    (0x7bff, 0xfbff, 0x7c00, 0xfc00),  # greatest finite values and infinities
    (0x3555, 0xb555, 0x3c01, 0xbc01),  # normal values with fraction bits
)


def hex_texel(halves):
    values = struct.unpack('<4e', struct.pack('<4H', *halves))
    return ','.join('%08x' % word for word in struct.unpack('<4I', struct.pack('<4f', *values)))


def translated(args, directory, halves):
    source = fixture.D16_SOURCE
    for previous, pair in zip(('0xc0003c00', '0x7bff3800'), (halves[:2], halves[2:])):
        assert source.count(previous) == 1
        source = source.replace(previous, hex(pair[0] | (pair[1] << 16)))
    asm, obj, raw = (directory / name for name in ('guest.s', 'guest.o', 'guest.bin'))
    asm.write_text(source, encoding='ascii')
    subprocess.run([args.clang, '-cc1as', '-triple', 'amdgcn--amdpal', '-target-cpu', 'gfx1013',
                    '-target-feature', '+wavefrontsize32', '-filetype', 'obj', asm, '-o', obj], check=True)
    subprocess.run([args.objcopy, '-O', 'binary', '--only-section=.text', obj, raw], check=True)
    hlsl = fixture.translate(raw.read_bytes()).hlsl_ps()
    assert hlsl.count(fixture.D16_LINE) == 1
    return hlsl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('clang', 'objcopy', 'dxc', 'renderer', 'adapter'):
        parser.add_argument(name)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        vs = fixture.compile_hlsl(args.dxc, 'dxc', fixture.FULLSCREEN_VS, 'fullscreen', 'vs_6_0', directory)
        for vector in VECTORS:
            for rotation in range(4):
                halves = vector[rotation:] + vector[:rotation]
                expected = hex_texel(halves)
                source = translated(args, directory, halves)
                # The renderer sets both root constants to zero. Force conversions at draw time in this variant.
                runtime = source.replace(fixture.D16_LINE, 'v4 ^= ud0; v5 ^= ud1;\n    ' + fixture.D16_LINE)
                swapped = runtime.replace(fixture.D16_LINE, fixture.D16_LINE[:-1] + '.yxwz;')
                for label, shader in (('translated', source), ('runtime', runtime), ('swapped', swapped)):
                    ps = fixture.compile_hlsl(args.dxc, 'dxc', shader, label, 'ps_6_0', directory)
                    if label == 'runtime':
                        dump = subprocess.run([args.dxc, '-dumpbin', ps], capture_output=True, text=True, check=True)
                        assert dump.stdout.count('call float @dx.op.legacyF16ToF32') == 4, expected
                    run = subprocess.run([args.renderer, args.adapter, '--startup-image-store', vs, ps,
                                          '1,2', expected], capture_output=True, text=True)
                    if run.returncode == 77:
                        raise SystemExit(77)
                    if label == 'swapped':
                        actual = hex_texel([halves[i] for i in (1, 0, 3, 2)])
                        diagnostic = f'texel_mismatch x=1 y=2 expected={expected} actual={actual}'
                        assert run.returncode == 1 and diagnostic in run.stderr, (label, expected, run.stderr)
                    else:
                        assert run.returncode == 0, (label, expected, run.stderr, run.stdout)
                print('passed translated, runtime, swapped', expected, flush=True)


if __name__ == '__main__':
    main()
