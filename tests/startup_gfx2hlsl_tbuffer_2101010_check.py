"""Packed 2-10-10-10 UNORM typed buffer loads in tools/startup/gfx2hlsl.py (Part of #1723).

usage: startup_gfx2hlsl_tbuffer_2101010_check.py

`BUF_FMT_2_10_10_10_UNORM` stores x, y and z as 10-bit fields and w as a 2-bit field in one dword, lowest bits
first. The format table did not name it, so a typed load of it was refused with
`buffer format BUF_FMT_2_10_10_10_UNORM`. DXGI_FORMAT_R10G10B10A2_UNORM (24) in an input layout maps to it.

The instruction bytes are constructed here from the field layout the checked-in decoder reads
(`tools/disasm_rdna2.py`, the MTBUF branch): no game-derived bytes are used. Each case encodes the instruction,
decodes it, translates it, and then evaluates the emitted component expressions against a reference that
decodes the same dword from the format definition (UNORM n / (2^bits - 1)).
"""
import os
import re
import struct
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'tools'))
sys.path.insert(0, os.path.join(here, '..', 'tools', 'startup'))
import gfx2hlsl as G  # noqa: E402

failures = []


def check(label, condition, detail=''):
    print(f"{label}: {'ok' if condition else 'MISMATCH'}")
    if not condition:
        failures.append(label)
        if detail:
            print('  ' + detail)


MTBUF_LOAD = {'x': 0, 'xy': 1, 'xyz': 2, 'xyzw': 3}
MTBUF_FORMAT_2_10_10_10_UNORM = 50
MTBUF_FORMAT_2_10_10_10_SNORM = 51  # a sibling the table must keep refusing


def encode_mtbuf(op, fmt, *, vdata=0, vaddr=0, srsrc=4, soffset=128):
    """Encode one MTBUF instruction (idxen); returns the 8 bytes the decoder reads."""
    w0 = (0b111010 << 26) | ((fmt & 0x7f) << 19) | ((op & 0x7) << 16) | (1 << 13)
    w1 = (vaddr & 0xff) | ((vdata & 0xff) << 8) | (((srsrc // 4) & 0x1f) << 16) | (((op >> 3) & 1) << 21) | (soffset << 24)
    return struct.pack('<II', w0, w1)


class Pipeline:
    def __init__(self):
        self.resources, self.notes, self.modified = {}, set(), set()

    def resource(self, key, kind):
        entry = self.resources.setdefault(key, dict(index=len(self.resources), kind=kind))
        return 'r%d' % entry['index']

    def ud(self, source):
        return 0


class Stage(G.Stage):
    def __init__(self):
        self.lines, self.sym = [], {}
        self.used_s, self.used_v, self.uniform_v = set(), set(), set()
        self.exec_known = True
        self.exports = {}
        self.pipe = Pipeline()

    def descriptor(self, tok):
        return ('buffer', tok)


def translate_bytes(code):
    decoded = G.decode(code)
    stage = Stage()
    for _offset, _size, name, asm in decoded:
        stage.one(name, asm)
    return [asm for _o, _s, _n, asm in decoded], stage.lines


def run_components(lines, word):
    """Evaluate the emitted `vN = <expr>;` lines with `w` bound to the loaded dword; returns {N: value}."""
    out = {}
    for line in lines:
        m = re.fullmatch(r'v(\d+) = (.*);', line.strip())
        if m:
            out[int(m.group(1))] = eval(m.group(2), {'U': lambda x: x, 'ldu': lambda *_: 0}, {'w': word})
    return out


def reference(word):
    return [(word & 1023) / 1023.0, ((word >> 10) & 1023) / 1023.0, ((word >> 20) & 1023) / 1023.0, (word >> 30) / 3.0]


code = encode_mtbuf(MTBUF_LOAD['xyzw'], MTBUF_FORMAT_2_10_10_10_UNORM, vdata=1, vaddr=2, srsrc=4)
asm, lines = translate_bytes(code)
check('decodes the named format', 'BUF_FMT_2_10_10_10_UNORM' in asm[0], repr(asm))
check('the word is loaded with ldu', any('w = ldu(r0, a)' in l for l in lines), repr(lines))

for word in (0x00000000, 0xffffffff, 0x4012a3ff, 0x8007fc01, 0xc0000000, 0x2aaaaaaa, 0x155ff3d1):
    got = run_components(lines, word)
    want = reference(word)
    check('all four components for 0x%08x' % word, [got.get(1 + i) for i in range(4)] == want,
          'got %r want %r' % (got, want))

# Fewer components read the same leading fields.
_, lines_xy = translate_bytes(encode_mtbuf(MTBUF_LOAD['xy'], MTBUF_FORMAT_2_10_10_10_UNORM, vdata=1, vaddr=2))
got = run_components(lines_xy, 0x4012a3ff)
check('the xy form reads two components', sorted(got) == [1, 2] and [got[1], got[2]] == reference(0x4012a3ff)[:2], repr(got))

# The input-layout mapping: DXGI_FORMAT_R10G10B10A2_UNORM is 24, one dword per element.
check('DXGI 24 maps to the format', G.FETCH_DXGI.get(24) == ('BUF_FMT_2_10_10_10_UNORM', 4), repr(G.FETCH_DXGI.get(24)))

# A sibling format stays refused.
try:
    translate_bytes(encode_mtbuf(MTBUF_LOAD['xyzw'], MTBUF_FORMAT_2_10_10_10_SNORM, vdata=1, vaddr=2))
    refused = False
except G.Unsupported:
    refused = True
check('the SNORM sibling stays refused', refused)

if failures:
    print('FAIL')
    for text in failures:
        print('  ' + text)
    sys.exit(1)
print('PASS')
