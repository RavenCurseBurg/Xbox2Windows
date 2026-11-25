"""Compare direct and translated integer PS exports on four-channel 8-bit render targets."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from unittest.mock import patch

import startup_mrt_check as fixture


SETS = (
    (0xffffffff, 0xffffff80, 0x00000080, 0x00000100),
    (0xffffff7f, 0x0000007f, 0x00000002, 0x000000fe),
    (0x00000000, 0x00000001, 0x0000007f, 0x00000042),
)


def measure(directory, consumer, backend, kind, dxc, fxc):
    gfx = fixture.gfx2hlsl
    cache = directory / 'cache'
    cache.mkdir()
    vs, ps, root, out = (directory / n for n in ('vs', 'ps', 'root', 'pixels'))
    vs.write_bytes(fixture.vertex())
    env = dict(os.environ, X2W_D3D12_PERMISSIVE='1', X2W_D3D12_LIST_NOOP='1', X2W_D3D12_RENDER='1',
               X2W_D3D12_DEBUG='1', X2W_SHADER_CACHE=str(cache))
    compiler = fxc if kind == 'fxc' else dxc

    def draw(fmt, expected_draw):
        proc = subprocess.run([consumer, vs, ps, root, backend, out, str(fmt), '0'], env=env,
                              capture_output=True, text=True)
        if proc.returncode == 77:
            raise SystemExit(77)
        assert proc.returncode == 0 and ('-> drawn;' in proc.stderr) == expected_draw, proc.stderr
        assert not re.search(r'x2w-d3d12-debug.*sev=[012]\b', proc.stderr), proc.stderr
        adapter = next(s for s in proc.stdout.splitlines() if s.startswith('adapter='))
        debug = next(s for s in proc.stderr.splitlines() if 'debug layer:' in s)
        return out.read_bytes(), adapter, debug

    for words in SETS:
        ps.write_bytes(fixture.pixel(words, unorm=True))
        for fmt, hlsl_type in ((30, 'uint'), (32, 'int')):
            cleared, adapter, debug = draw(fmt, False)
            assert cleared == bytes.fromhex('000001010000ffff'), (fmt, cleared.hex())
            base = gfx.pipeline_key(vs.read_bytes(), ps.read_bytes(), root.read_bytes())
            record = f'rtv1 {base} {fmt} 28'
            key = gfx.fnv(record.encode('ascii'))
            name = f'{key:016x}'
            assert (cache / f'req_{name}.rtv').read_text() == record
            # This experiment widens only the test process's verified set.
            with patch.object(gfx, 'VERIFIED_UINT_RTVS', gfx.VERIFIED_UINT_RTVS | {30}), \
                 patch.object(gfx, 'VERIFIED_SINT_RTVS', gfx.VERIFIED_SINT_RTVS | {32}), \
                 patch.dict(os.environ, {'X2W_FXC': ''}):
                status = gfx.translate_request(cache, key)
            assert (cache / f'{name}.txt').read_text().splitlines()[-1] == record, status
            for stage in ('vs', 'ps'):
                fixture.compile_shader(compiler, kind, stage, cache / f'{name}.{stage}.hlsl',
                                       cache / f'{name}.{stage}.cso', cache / f'{name}.{stage}.asm')
            translated, _, _ = draw(fmt, True)
            literal = ', '.join(f'0x{word:08x}u' for word in words)
            value = f'uint4({literal})'
            if hlsl_type == 'int':
                value = f'asint({value})'
            direct = directory / 'direct.hlsl'
            direct.write_text(f'struct Outputs {{ {hlsl_type}4 a : SV_Target0; float4 b : SV_Target1; }};\n'
                              f'Outputs main() {{ Outputs o; o.a = {value}; o.b = float4(1,0,0,1); return o; }}\n')
            fixture.compile_shader(compiler, kind, 'ps', direct, cache / f'{name}.ps.cso', directory / 'direct.asm')
            reference, _, _ = draw(fmt, True)
            signed = [w - (1 << 32) if w >= (1 << 31) else w for w in words]
            saturated = (bytes(max(-128, min(127, n)) & 255 for n in signed) if fmt == 32
                         else bytes(min(255, w) for w in words))
            low = bytes(w & 255 for w in words)
            print('RGBA8_RESULT', json.dumps(dict(backend=backend, compiler=kind, format=fmt, adapter=adapter,
                  debug=debug, words=[f'{w:08x}' for w in words], translated=translated.hex(), direct=reference.hex(),
                  clamp=saturated.hex(), low_bytes=low.hex())), flush=True)
            assert len(reference) == 8 and reference[4:] == bytes.fromhex('ff0000ff') and translated == reference


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as work:
        measure(Path(work), *sys.argv[1:])
