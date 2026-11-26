"""Assemble conditional copies with unequal interpolated halves and both destination/source aliases."""
import argparse
from pathlib import Path
import subprocess
import tempfile

import startup_gfx2hlsl_interp_f16_check as F


def assemble(asm, args, root):
    src,obj,raw = (root/n for n in ('pixel.s','pixel.o','pixel.bin'))
    src.write_text(asm)
    subprocess.run([args.clang,'-cc1as','-triple','amdgcn--amdpal','-target-cpu','gfx1013',
                    '-target-feature','+wavefrontsize32','-filetype','obj',src,'-o',obj],check=True)
    subprocess.run([args.objcopy,'-O','binary','--only-section=.text',obj,raw],check=True)
    return raw.read_bytes()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('clang','objcopy','dxc','renderer','adapter'):
        cli.add_argument(key)
    args = cli.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        vs = F.compile_hlsl(args.dxc,'dxc',F.pair()[0],'vertex','vs_6_0',root)
        values = {'v4': 1.5, 'v5': -2.0, 'v10': 0.5}
        for first, second in [('v4', 'v5'), ('v5', 'v4'), ('v4', 'v10'), ('v10', 'v4')]:
            for destination in (first, second, 'v8'):
                for truth in (False, True):
                    # The interpolated v6 low half is 0.5, so 0 > v9 is false and 1 > v9 is true.
                    head = (F.V_MIX_HEAD + 'v_mov_b32 v10, 0x40003800\nv_cvt_f32_f16 v9, v6\n'
                            + f'v_cmp_gt_f32_e32 vcc_lo, {int(truth)}.0, v9\n'
                            + f'v_cndmask_b32 {destination}, {first}, {second}, vcc_lo\n')
                    text = (head + f'v_cvt_f32_f16 v8, {destination}\n'
                            + 'v_cvt_f32_f16 v6, v6\nv_cvt_f32_f16 v7, v7\n'
                            + 'exp mrt0 v8, v9, v6, v7 done vm\ns_endpgm\n')
                    bad = text.replace(f'v_cvt_f32_f16 v8, {destination}', f'v_lshrrev_b32 v8, 16, {destination}')
                    try:
                        F.translate(assemble(bad,args,root))
                    except F.G.Unsupported as error:
                        assert 'upper 16 bits' in str(error), error
                    else:
                        raise AssertionError(('upper-half read accepted',destination,first,second))
                    ps = F.compile_hlsl(args.dxc,'dxc',F.translate(assemble(text,args,root)).hlsl_ps(),'pixel','ps_6_0',root)
                    for flip in (False,True):
                        selected = second if truth ^ flip else first
                        expected = (values[selected], 0.5, 0.5, 3.0)
                        want = ','.join('%08x' % F.f32_bits(v) for v in expected)
                        run = F.draw(args.renderer,args.adapter,vs,ps,want)
                        assert run.returncode == int(flip), (truth,destination,first,second,run.stderr,run.stdout)
                        if flip:
                            assert 'mismatch' in run.stderr.lower(),run.stderr
                    print(first,second,destination,truth,'passed selection, reversed reference and refusal',flush=True)


if __name__ == '__main__':
    main()
