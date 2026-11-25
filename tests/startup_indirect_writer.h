#pragma once
#include "desktop_gpu.h"

template <typename T>
Microsoft::WRL::ComPtr<T> indirect_native_object(ID3D12Device* device, ID3D12Device* guest, IUnknown* object)
{
    // SetPrivateDataInterface unwraps a guest argument before the call reaches the shared desktop device.
    constexpr GUID key{0x10a62244, 0x1615, 0x4953, {0xa2, 0x52, 0xbe, 0x42, 0x35, 0x27, 0xa0, 0x61}};
    check(guest->SetPrivateDataInterface(key, object), "publish underlying object");
    Microsoft::WRL::ComPtr<IUnknown> stored;
    UINT bytes = sizeof(IUnknown*);
    const HRESULT read = device->GetPrivateData(key, &bytes, stored.GetAddressOf());
    check(guest->SetPrivateDataInterface(key, nullptr), "clear underlying object");
    check(read, "retrieve underlying object");
    Microsoft::WRL::ComPtr<T> result;
    check(stored.As(&result), "query underlying object");
    return result;
}

inline void record_indirect_writer(ID3D12GraphicsCommandList* list, ID3D12PipelineState* writer,
                                   ID3D12Resource* arguments)
{
    list->SetPipelineState(writer);
    list->SetComputeRootUnorderedAccessView(0, arguments->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
}
