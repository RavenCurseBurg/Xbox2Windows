#include "bvh_fixture.h"
#include <dxgi1_6.h>
#include <fstream>
#include <iterator>
#include <string>
#include <cwchar>

using namespace bvh_fixture;
using Row4 = std::array<Word, 4>;

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const std::string backend = argv[1], shader_name = argv[2];
    const std::string log = backend + "-" + shader_name + ".receipt";
    if (!std::freopen(log.c_str(), "w", stdout)) return 3;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool mutant = shader_name == "first-active";
    try {
        ComPtr<IDXGIFactory4> factory;
        check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "factory");
        ComPtr<IDXGIAdapter1> adapter;
        if (backend == "WARP") {
            check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "warp");
        } else {
            const wchar_t* needle = backend == "NVIDIA" ? L"NVIDIA" : backend == "Intel" ? L"Intel" : nullptr;
            if (!needle) return 2;
            for (UINT i = 0; factory->EnumAdapters1(i, adapter.ReleaseAndGetAddressOf()) == S_OK; ++i) {
                DXGI_ADAPTER_DESC1 desc{};
                check(adapter->GetDesc1(&desc), "adapter description");
                if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) && std::wcsstr(desc.Description, needle)) break;
            }
            if (!adapter) return 77;
        }
        DXGI_ADAPTER_DESC1 desc{};
        check(adapter->GetDesc1(&desc), "chosen adapter");
        std::printf("%s vendor=%04x device=%04x shader=%s\n", backend.c_str(), desc.VendorId, desc.DeviceId, argv[2]);
        ComPtr<ID3D12Device> owner;
        check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&owner)), "device");
        auto* device = owner.Get();
        std::ifstream f(shader_name + ".cso", std::ios::binary);
        std::vector<char> shader{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>{}};
        if (shader.empty()) return 2;
        Kernel kernel(device, {shader.data(), shader.size()});
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = 4;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> heap;
        check(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap)), "descriptor table");
        auto handle = heap->GetCPUDescriptorHandleForHeapStart();
        std::array<ComPtr<ID3D12Resource>, 4> sources;
        for (Word i = 0; i < 4; ++i) {
            std::vector<Word> data;
            for (Word lane = 0; lane < 4; ++lane) data.push_back(1000 * (i + 1) + 17 * lane);
            sources[i] = upload(device, data);
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R32_TYPELESS;
            srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Buffer.NumElements = 4;
            srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            device->CreateShaderResourceView(sources[i].Get(), &srv, handle);
            handle.ptr += device->GetDescriptorHandleIncrementSize(heap_desc.Type);
        }
        const std::array<Row4, 5> inputs = {{{0,1,2,3},{3,2,1,0},{2,1,2,3},{4,9,14,19},{7,7,7,7}}};
        constexpr Word sentinel = 0xc001cafe;
        const std::vector<Word> blank(320, sentinel);
        auto initial = upload(device, blank);
        unsigned cases = 0, bad_cases = 0;
        for (const auto& input : inputs) {
            auto keys = upload(device, std::vector<Word>(input.begin(), input.end()));
            for (Word active = 0; active <= 4; ++active) {
                auto output = make_buffer(device, 1280, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                                          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
                auto back = make_buffer(device, 1280, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
                ComputeCommands cmd(device, kernel.pipeline.Get());
                auto* list = cmd.list.Get();
                list->CopyBufferRegion(output.Get(), 0, initial.Get(), 0, 1280);
                transition(list, output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                ID3D12DescriptorHeap* heaps[] = {heap.Get()};
                list->SetDescriptorHeaps(1, heaps);
                list->SetComputeRootSignature(kernel.root.Get());
                list->SetComputeRootShaderResourceView(0, keys->GetGPUVirtualAddress());
                list->SetComputeRootDescriptorTable(1, heap->GetGPUDescriptorHandleForHeapStart());
                list->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
                list->SetComputeRoot32BitConstant(3, active, 0);
                list->Dispatch(1,1,1);
                transition(list, output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->CopyBufferRegion(back.Get(), 0, output.Get(), 0, 1280);
                cmd.submit_and_wait(device);
                auto actual = read(back.Get(), 320);
                bool differs = false;
                for (Word j = 0; j < 320; ++j) {
                    const auto value = [&](Word index) { return 1000 * (1 + index % 4) + 17 * j; };
                    Word correct = j >= 4 ? sentinel : j < active ? value(input[j]) : 13;
                    Word expected = mutant && j < active ? value(input[0]) : correct;
                    if (actual[j] != expected) {
                        std::printf("case=%u word=%u expected=%08x actual=%08x\n", cases, j, expected, actual[j]);
                        return 1;
                    }
                    differs |= actual[j] != correct;
                }
                bad_cases += differs;
                ++cases;
            }
        }
        std::printf("cases=%u words/case=320 cases_different_from_per_lane=%u\n", cases, bad_cases);
        std::fflush(stdout);
        return cases == 25 && bad_cases == (mutant ? 12u : 0u) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}