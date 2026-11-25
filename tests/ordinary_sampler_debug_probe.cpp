// Independent verification harness for the D3D12 debug layer on WARP.
#include "startup_stubs/translated_sampler_bindings.h"
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <vector>

extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion = 620;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}

int main()
{
    using Microsoft::WRL::ComPtr;
    ComPtr<ID3D12Debug> debug;
    if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) return 77;
    debug->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> warp;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> queue;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) ||
        FAILED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) ||
        FAILED(device.As(&queue))) return 2;

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    heap_desc.NumDescriptors = 1;
    ComPtr<ID3D12DescriptorHeap> heap;
    if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)))) return 3;
    auto check = [&](const char* label, const D3D12_SAMPLER_DESC& desc, unsigned expected) {
        queue->ClearStoredMessages();
        device->CreateSampler(&desc, heap->GetCPUDescriptorHandleForHeapStart());
        unsigned matches = 0, errors = 0;
        for (UINT64 i = 0; i < queue->GetNumStoredMessagesAllowedByRetrievalFilter(); ++i) {
            SIZE_T bytes = 0;
            if (FAILED(queue->GetMessage(i, nullptr, &bytes))) return false;
            std::vector<BYTE> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(queue->GetMessage(i, message, &bytes))) return false;
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) ++errors;
            if (message->ID == 1361) ++matches;
        }
        std::printf("%s: warning_1361=%u errors=%u\n", label, matches, errors);
        return matches == expected && errors == 0;
    };
    auto ordinary = x2w_startup::dynamic_sampler({});
    auto old = ordinary;
    old.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    auto comparison = old;
    comparison.Filter = D3D12_FILTER_COMPARISON_MIN_MAG_MIP_POINT;
    const bool restored = check("restored ordinary ALWAYS", old, 1);
    const bool fixed = check("ordinary", ordinary, 0);
    const bool fallback = check("border fallback", x2w_startup::border_sampler(), 0);
    const bool actual_comparison = check("comparison ALWAYS", comparison, 0);
    return restored && fixed && fallback && actual_comparison ? 0 : 1;
}
