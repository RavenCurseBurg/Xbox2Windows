"""Compare translated and independent integer exports, destination packing and low-bit controls."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

import startup_mrt_check as fixture


FORMATS = (
    (12, 'uint', (16, 16, 16, 16)), (14, 'int', (16, 16, 16, 16)),
    (25, 'uint', (10, 10, 10, 2)),
    (36, 'uint', (16, 16)), (38, 'int', (16, 16)),
    (50, 'uint', (8, 8)), (52, 'int', (8, 8)),
    (57, 'uint', (16,)), (59, 'int', (16,)),
    (62, 'uint', (8,)), (64, 'int', (8,)),
)
SETS = (
    (0x00010001, 0xfffefffe, 0x0000ffff, 0x00000104),
    (0xffff7ffe, 0x00010000, 0x00007f01, 0xffffff82),
    (0x00000021, 0x00000012, 0x00000035, 0x00000002),
)


def signed(word, bits):
    return word - (1 << bits) if word & (1 << (bits - 1)) else word


def packed(words, widths, hlsl_type, truncate=False):
    values = list(words)
    shift = pixel = 0
    for index, (word, width) in enumerate(zip(words, widths)):
        mask = (1 << width) - 1
        if truncate:
            n = word & mask
            if hlsl_type == 'int':
                n = signed(n, width)
        elif hlsl_type == 'int':
            n = min(mask >> 1, max(-(1 << (width - 1)), signed(word, 32)))
        else:
            n = min(word, mask)
        values[index] = n & 0xffffffff
        pixel |= (n & mask) << shift
        shift += width
    return pixel.to_bytes(shift // 8, 'little'), values


def measure(directory, consumer, backend, compiler_kind, dxc, fxc):
    gfx = fixture.gfx2hlsl
    cache = directory / 'cache'
    cache.mkdir()
    vs, ps, root, out = (directory / n for n in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(fixture.vertex())
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache))
    compiler = fxc if compiler_kind == 'fxc' else dxc

    def draw(fmt, wanted):
        result = subprocess.run([consumer, vs, ps, root, backend, out, str(fmt), '0'], env=env,
                                capture_output=True, text=True)
        if result.returncode == 77:
            raise SystemExit(77)
        assert result.returncode == 0 and ('-> drawn;' in result.stderr) == wanted, (fmt, result.stderr)
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', result.stderr), result.stderr
        adapter = next(s for s in result.stdout.splitlines() if s.startswith('adapter='))
        debug = next(s for s in result.stderr.splitlines() if 'debug layer:' in s)
        return out.read_bytes(), adapter, debug

    def direct(fmt, words, hlsl_type, destination):
        literal = ', '.join(f'0x{word:08x}u' for word in words)
        value = f'uint4({literal})'
        if hlsl_type == 'int':
            value = f'asint({value})'
        source = directory / 'direct.hlsl'
        source.write_text(f'struct Outputs {{ {hlsl_type}4 a : SV_Target0; float4 b : SV_Target1; }};\n'
                          f'Outputs main() {{ Outputs o; o.a = {value}; o.b = float4(1,0,0,1); return o; }}\n')
        fixture.compile_shader(compiler, compiler_kind, 'ps', source, destination, directory / 'direct.asm')
        return draw(fmt, True)[0]

    for words in SETS:
        ps.write_bytes(fixture.pixel(words, unorm=True))
        for fmt, hlsl_type, widths in FORMATS:
            clear, adapter, debug = draw(fmt, False)
            assert clear == packed((0, 0, 1, 1), widths, hlsl_type)[0] + bytes.fromhex('0000ffff'), (fmt, clear.hex())
            base = gfx.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
            record = f'rtv1 {base} {fmt} 28'
            key = gfx.fnv(record.encode('ascii'))
            name = f'{key:016x}'
            assert (cache / f'req_{name}.rtv').read_text() == record
            verified = 'VERIFIED_UINT_RTVS' if hlsl_type == 'uint' else 'VERIFIED_SINT_RTVS'
            with patch.object(gfx, verified, getattr(gfx, verified) | {fmt}), patch.dict(os.environ, {'X2W_FXC': ''}):
                status = gfx.translate_request(cache, key)
            assert (cache / f'{name}.txt').read_text().splitlines()[-1] == record, status
            for stage in ('vs', 'ps'):
                fixture.compile_shader(compiler, compiler_kind, stage, cache / f'{name}.{stage}.hlsl',
                                       cache / f'{name}.{stage}.cso', cache / f'{name}.{stage}.asm')
            assert fixture.signature(cache / f'{name}.ps.asm') == {'0': hlsl_type, '1': 'float'}
            translated = draw(fmt, True)[0]
            saturated, _ = packed(words, widths, hlsl_type)
            low, low_values = packed(words, widths, hlsl_type, truncate=True)
            reference = direct(fmt, words, hlsl_type, cache / f'{name}.ps.cso')
            control = direct(fmt, low_values, hlsl_type, cache / f'{name}.ps.cso')
            other = bytes.fromhex('ff0000ff')
            assert translated == reference == saturated + other, (fmt, translated.hex(), reference.hex(), saturated.hex())
            assert control == low + other, (fmt, control.hex(), low.hex())
            assert (saturated == low) == (words == SETS[-1]), (fmt, words)
            print('NARROW_RESULT', json.dumps(dict(backend=backend, compiler=compiler_kind, format=fmt,
                  words=[f'{word:08x}' for word in words], adapter=adapter, debug=debug, translated=translated.hex(),
                  direct=reference.hex(), truncated=control.hex())), flush=True)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        measure(Path(tmp), *sys.argv[1:])
