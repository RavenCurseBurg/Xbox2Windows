"""Compare startup and direct 2D GET_LOD using the explicit-SRV measurement renderer."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile

import startup_gfx2hlsl_image_get_lod_check as fixture
from startup_flat_param_check import assemble
from startup_image_3d_check import FULLSCREEN_VS, compile_hlsl

# MostDetailedMip / MipLevels, coordinate step per screen pixel, guest sampler limits.
CASES = (
    ('0/3', 1, 0, 2), ('0/3', 1, 0, .75), ('0/3', .125, .5, 2),
    ('0/3', .5, 1.5, 2), ('1/2', 1, 0, 2), ('1/2', 1, 0, .75),
    ('1/2', .25, 0, 2), ('1/1', 1, 0, 2), ('0/1', 1, 0, 2),
)

def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]

def translated(llvm, step, low, high, mip_filter):
    stage = fixture.stage(dmask=3, word2={'point': 1, 'linear': 2}[mip_filter] << 26)
    limit = int(low * 256) | int(high * 256) << 12
    limits = assemble(llvm, f's_mov_b32 s25, 0x{limit:x}\n')
    coords = assemble(llvm, f'''v_lshrrev_b32 v1, 16, v0
v_and_b32 v0, 0xffff, v0
v_cvt_f32_u32 v0, v0
v_cvt_f32_u32 v1, v1
v_mul_f32 v0, 0x{bits(step):08x}, v0
v_mul_f32 v1, 0x{bits(step):08x}, v1
''')
    code = stage[:16] + limits + stage[20:32] + coords + stage[52:]
    pipeline = fixture.G.Pipeline(fixture.VS_WORDS, code, fixture.root_blob(flags=0), ps_hw_inputs=(0x8000,0x8000))
    manifest = next(line for line in pipeline.manifest().splitlines() if line.startswith('smpl '))
    return pipeline.hlsl_ps(), manifest

def direct(step):
    return f'''Texture2D<float4> image : register(t0);
SamplerState sampler0 : register(s0);
float4 main(float4 position : SV_Position) : SV_Target0 {{
    float2 coord = floor(position.xy) * {float(step)};
    float x = image.CalculateLevelOfDetail(sampler0, coord);
    float y = image.CalculateLevelOfDetailUnclamped(sampler0, coord);
    return float4(x, y, x, y);
}}
'''

def draw(renderer, vs, ps, manifest, view, adapter, expected):
    env = dict(os.environ, X2W_GET_LOD_VIEW=view, XBOX2WINDOWS_ADAPTER=adapter)
    backend = 'warp' if adapter == 'warp' else 'hardware'
    wanted = ','.join(f'{bits(x):08x}' for x in (expected * 2))
    result = subprocess.run([renderer, backend, '--startup-image-2d', vs, ps, '0', wanted, manifest],
                            env=env, capture_output=True, text=True, timeout=60)
    row = re.search(r'lod_query resource_base=4 resource_mips=3 view_first=(\d+) view_levels=(\d+) sampler_min=(\S+) sampler_max=(\S+)',result.stdout)
    device = re.search(r'adapter=(.*) vendor=(\w+) device=(\w+)\s+driver=(\S+)',result.stdout)
    if not (row and device and (int(row[1]), int(row[2])) == tuple(map(int, view.split('/')))):
        raise AssertionError(result.stdout + result.stderr)
    actual = wanted
    if result.returncode == 1:
        mismatch = re.search(r'startup_image mismatch .* expected=(\S+) actual=([0-9a-f,]+)', result.stderr)
        if not mismatch or mismatch[1] != wanted:
            raise AssertionError(result.stdout + result.stderr)
        actual = mismatch[2]
    elif result.returncode != 0:
        raise AssertionError(result.stdout + result.stderr)
    color = tuple(struct.unpack('>f', bytes.fromhex(word))[0] for word in actual.split(','))
    assert color[:2] == color[2:], color
    return color[:2], [float(row[3]),float(row[4])], device[1], device[4], result.returncode

def main(dxc, renderer, llvm, adapter):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        vs = compile_hlsl(dxc,'dxc',FULLSCREEN_VS,'full','vs_6_0',work)
        cases = [('point', *case) for case in CASES]
        cases.extend(('linear', *case) for case in CASES[1:4])
        for i,(mip_filter,view,step,low,high) in enumerate(cases):
            code, line = translated(llvm,step,low,high,mip_filter)
            ps = compile_hlsl(dxc,'dxc',code,f'guest{i}','ps_6_0',work)
            other = compile_hlsl(dxc,'dxc',direct(step),f'direct{i}','ps_6_0',work)
            first,count = map(int,view.split('/'))
            raw = math.log2(step*(4 >> first))
            clamp = max(0,min(count-1,max(low,min(high,raw))))
            probe = draw(renderer,vs,ps,line,view,adapter,(clamp,raw))
            reference = draw(renderer,vs,other,line,view,adapter,(clamp,raw))
            assert probe == reference and probe[1] == [low,high], (probe,reference)
            print(json.dumps(dict(mip_filter=mip_filter,view=view,step=step,sampler=probe[1],model=[clamp,raw],
                                  measured=probe[0],device=probe[2],driver=probe[3],strict_exit=probe[4])),flush=True)
            if i == 1:
                dropped = draw(renderer,vs,ps,line.replace('smpl ', 'smp ', 1),view,adapter,(clamp,raw))
                assert dropped[1][0] == 0 and dropped[1][1] > 2
                assert dropped[0] == (2.0, 2.0) and dropped[0] != probe[0], (dropped, probe)
                print(json.dumps(dict(control='drop-sampler-limits',view=view,measured=dropped[0],original=probe[0],device=dropped[2])),flush=True)

if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('dxc','renderer','llvm','adapter'):
        cli.add_argument(name)
    main(**vars(cli.parse_args()))
