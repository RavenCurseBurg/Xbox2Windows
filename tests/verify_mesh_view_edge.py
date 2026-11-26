"""Draw a flat-coloured triangle whose provoking vertex reads the last 1, 2, 3 or 4 words of the 512-byte view."""
import argparse
import struct
import startup_mesh57_buffer_read_check as fixture


def image(start):
    colours = []
    for slot in range(len(fixture.POINTS)):
        channels = []
        for channel in range(4):
            offset = start + slot*16 + channel*4
            channels.append(0 if offset >= 512 else round(struct.unpack_from('<f', fixture.VIEW, offset)[0]*255))
        colours.append(tuple(channels))
    return fixture.T.rasterise(fixture.POINTS,colours,fixture.INDICES,len(colours),len(fixture.INDICES))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dxc','renderer','adapter'):
        p.add_argument(key)
    args = p.parse_args()
    view = bytearray(fixture.VIEW)
    for offset in range(448,512,4):
        struct.pack_into('<f',view,offset,(33+(offset-448)*3)/255)
    fixture.VIEW = bytes(view)
    # Triangle 2 takes its flat colour from slot 3. The other provoking slots are all beyond it.
    assert fixture.INDICES[2][0] == 3
    cases = []
    for address in [496,500,504,508,512]:
        start = address-3*16
        drawn, shifted = image(start),image(start+4 if address != 512 else start-4)
        assert shifted != drawn
        guest = fixture.soffset_variant(start)
        cases += [(f'first byte {address}',guest,None,drawn,drawn),
                  (f'first byte {address} wrong address',guest,None,drawn,shifted)]
    fixture.CASES = cases
    return fixture.draw(args.dxc,args.renderer,args.adapter)


if __name__ == '__main__':
    raise SystemExit(main())
