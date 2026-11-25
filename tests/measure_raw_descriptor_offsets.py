"""Reassemble the #2069/#2546 compute cases and check their descriptor bindings. Argument: llvm-mc."""
import sys

import startup_gfx2hlsl_compute_check as cases


def source(name, step, immediate):
    original = 's_load_dwordx8 s[12:19], s[10:11], 0x0'
    first = f's_load_dwordx8 s[12:19], s[10:11], {immediate:#x}'
    if step:
        first = f's_add_u32 s10, s10, {step}\n' + first
    text = cases.CASES[name][0]
    assert text.count(original) == 1
    return text.replace(original, first)


def translated(llvm, name, text):
    return cases.translate(name, code=cases.assemble(llvm, text))


def verify(llvm):
    for step, immediate in ((0, 0), (16, 0), (16, 16), (32, 0), (32, 16), (64, 16)):
        total = step + immediate
        raw_source = source('I-raw', step, immediate)
        plain = translated(llvm, 'A', source('A', step, immediate))
        raw = translated(llvm, 'I-raw', raw_source)
        assert raw.hlsl_cs() == plain.hlsl_cs()
        for program in (plain, raw):
            bindings = [s for s in program.manifest().splitlines() if s.startswith(('srv ', 'uav '))]
            assert bindings == [f'srv 0 buf table 1 {total}', f'uav 0 table 1 {total + 16}'], bindings

        # Reload through the same low word, but take the stride from a different high word.
        reload_stride = (f's_load_dwordx8 s[24:31], s[10:11], {immediate:#x}\n'
                         's_waitcnt lgkmcnt(0)\n'
                         's_bfe_u32 s21, s29, 0xe0010')
        if not step:
            same = raw_source.replace(cases.STRIDE_EXTRACT, reload_stride)
            assert translated(llvm, 'I-raw', same).hlsl_cs() == raw.hlsl_cs()
        wrong = raw_source.replace(cases.STRIDE_EXTRACT, 's_mov_b32 s11, 1\n' + reload_stride)
        try:
            translated(llvm, 'I-raw', wrong).hlsl_cs()
        except cases.G.Unsupported as error:
            assert 'buffer store through a modified descriptor' in str(error), str(error)
        else:
            raise AssertionError('mixed two different high words in the structured-to-raw conversion')
        print(f'add {step} + load {immediate}: SRV {total}, UAV {total + 16}; mixed high word refused')


if __name__ == '__main__':
    verify(sys.argv[1])
