#pragma once
#include <d3d12.h>
#include <dxgi.h>
#include <wrl/client.h>

// Keeps the producer, argument resource and signature alive until the consumer's queue fence.
struct Slot67Indirect {
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> writer;
    Microsoft::WRL::ComPtr<ID3D12Resource> placed, upload, readback;
    Microsoft::WRL::ComPtr<ID3D12CommandSignature> signature;

    void prepare(IDXGIAdapter* adapter, void* guest_device, void* guest_list, void* backing, UINT groups);
    void execute(void* guest_list, UINT commands, bool count_buffer, bool outside) const;
    bool check_record(UINT groups) const;
};
