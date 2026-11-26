ByteAddressBuffer keys : register(t0);
#if defined(STATIC_SWITCH)
ByteAddressBuffer r0 : register(t1), r1 : register(t2), r2 : register(t3), r3 : register(t4);
#else
ByteAddressBuffer records[4] : register(t1);
#endif
RWByteAddressBuffer result : register(u0);
cbuffer config : register(b0) { uint active; };

[RootSignature("SRV(t0), DescriptorTable(SRV(t1, numDescriptors=4)), UAV(u0), RootConstants(num32BitConstants=1,b0)")]
[numthreads(4, 1, 1)]
void main(uint3 tid : SV_GroupThreadID)
{
    uint lane = tid.x;
    uint value = 13;
    if (lane < active) {
#if defined(FIRST_ACTIVE)
        uint index = keys.Load(0) & 3;
#else
        uint index = keys.Load(4 * lane) & 3;
#endif
#if defined(STATIC_SWITCH)
        switch (index) {
        case 0: value = r0.Load(4 * lane); break;
        case 1: value = r1.Load(4 * lane); break;
        case 2: value = r2.Load(4 * lane); break;
        default: value = r3.Load(4 * lane); break;
        }
#elif defined(NO_MARKER)
        value = records[index].Load(4 * lane);
#else
        value = records[NonUniformResourceIndex(index)].Load(4 * lane);
#endif
    }
    result.Store(4 * lane, value);
}