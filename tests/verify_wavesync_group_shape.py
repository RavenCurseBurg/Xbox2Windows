"""Read all WaveSync outputs using an independent dispatch-thread index for the UAV address."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

from startup_gfx2hlsl_wavesync_check import G


def shader(shape, wave, wrong):
    ws = G.WaveSync(shape, wave)
    lines = ['cbuffer entry : register(b0) { uint rounds; };', 'RWByteAddressBuffer u0 : register(u0);',
             *ws.declarations(), '[numthreads(%d,%d,%d)]' % shape,
             'void main(uint3 tid : SV_GroupThreadID, uint id : SV_GroupIndex) {', *ws.prologue(),
             'for (uint p=0; p<rounds; ++p) {', f'uint lane = id % {wave}u;',
             f'uint last = {wave-1}u;',
             'bool active = p%4==0 ? lane==last : p%4==1 ? false : p%4==2 ? lane==0 : lane==2 || lane==9;',
             'uint value = id*37u + 1009u*p + 13u;']
    first, value, code = ws.gather_first('active', 'value')
    any_bit, any_code = ws.gather_any('p%4==1 ? lane==1 : p%4==2 ? lane==last : p%4==3 && lane==last/2u')
    pick = ('(p%2==0 ? last : 2u)' if not wrong else '(p%2==0 ? last-1u : 3u)')
    selected, read_code = ws.gather_lane('value', pick)
    lines += code + any_code + read_code
    lines += [f'if (p+1==rounds) u0.Store4(id*16u,uint4({first},{value},{any_bit},{selected}));', '}', '}']
    return '\n'.join(lines)


def expected(threads, wave, rounds, wrong=False):
    result = []
    p = rounds - 1
    active = ((wave-1,), (), (0,), (2, 9))[p % 4]
    index = ((wave-2 if wrong else wave-1) if p % 2 == 0 else (3 if wrong else 2))
    for thread in range(threads):
        base = thread // wave * wave
        live = [i for i in active if base + i < threads]
        marker = (None, 1, wave - 1, (wave - 1) // 2)[p % 4]
        def val(i):
            return (base + i) * 37 + 1009 * p + 13
        result += [min(live) if live else 0xffffffff, val(min(live)) if live else 0,
                   int(marker is not None and base + marker < threads), val(index) if base+index < threads else 0]
    return result + [0xc5c5c5c5]*(320-len(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dxc', 'renderer', 'adapter'):
        p.add_argument(key)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for shape in [(5,4,3), (7,5,2)]:
            threads = shape[0]*shape[1]*shape[2]
            for wave in [32,64]:
                for wrong in [False,True]:
                    src, obj, case = (root/f for f in ('cs.hlsl','cs.dxil','in.case'))
                    src.write_text(shader(shape,wave,wrong))
                    subprocess.run([args.dxc,'-T','cs_6_0','-E','main','-Fo',obj,src],check=True)
                    for rounds in [1,2,3,4,7]:
                        good = expected(threads,wave,rounds)
                        actual = expected(threads,wave,rounds,wrong)
                        for words in ([actual,good] if actual != good else [good]):
                            case.write_bytes(struct.pack('<I320I16I',0x010101,*words,rounds,*([0]*15)))
                            run = subprocess.run([args.renderer,args.adapter,'--compute-store-case',obj,case],
                                                 capture_output=True,text=True)
                            if run.returncode == 77:
                                raise SystemExit(77)
                            count = sum(a != b for a,b in zip(actual,words))
                            assert run.returncode == int(bool(count)), (shape,wave,rounds,run.stderr,run.stdout)
                            assert f'compared_words=320 mismatches={count}' in run.stdout, (count,run.stderr,run.stdout)
                        print(shape,wave,rounds,'wrong' if wrong else 'original',sum(a!=b for a,b in zip(actual,good)),flush=True)


if __name__ == '__main__':
    main()
