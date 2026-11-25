// Translated compute programs at Dispatch (list slot 14). A device slot 67 pipeline is a placeholder that keeps
// the key of its published request; gfx2hlsl.py --pending answers the request with <key>.txt (`stage cs`) and
// <key>.cs.cso. This first slice binds raw buffers only: `srv <i> buf table <param> <byte>` inputs and
// `uav <i> table <param> <byte>` outputs named by hardware buffer descriptors in the compute tables recorded by
// slot 35, with entry values from the compute root constants of slots 37 and 39. Any other manifest line makes
// the translation unavailable; a dispatch that cannot be bound completely is refused with one reason and records
// nothing on the desktop list.

struct ComputeTranslation {
    struct Slot { int a, b; };  // root parameter and byte offset (views), or parameter and word (entry values)
    u64 key = 0;
    int state = 0;                    // 0 not loaded, 1 ready, 2 unavailable
    ULONGLONG retry_manifest_at = 0;  // only an absent manifest can become ready later
    std::vector<Slot> uds, srvs, uavs;
    x2w_startup::ComPtr<ID3D12RootSignature> root;
    x2w_startup::ComPtr<ID3D12PipelineState> pipeline;
    ~ComputeTranslation()
    {
        // The per-process cache is destroyed during DLL detach. A driver may already have shut
        // down when the process is terminating. Explicit unload still releases both references.
        if (x2w_d3d12_process_terminating) {
            pipeline.Detach();
            root.Detach();
        }
    }
};
std::unordered_map<u64, ComputeTranslation>& compute_translations() { static std::unordered_map<u64, ComputeTranslation> m; return m; }
LONG compute_dispatches_translated, compute_dispatches_refused, compute_dispatches_untranslated;

// draw_lock held. Returns the reason the translation is unavailable, or null when it is ready or still absent.
const char* load_compute_translation(ComputeTranslation& tr)
{
    tr.state = 2;
    wchar_t path[MAX_PATH];
    std::string manifest;
    if (!cache_path(path, L"", tr.key, L".txt") || !read_text(path, manifest)) {
        tr.retry_manifest_at = GetTickCount64() + 1000;
        return nullptr;
    }
    tr.retry_manifest_at = 0;
    unsigned declared_uds = 0, line_number = 0;
    for (std::size_t at = 0; at < manifest.size(); ++line_number) {
        std::size_t end = manifest.find('\n', at);
        if (end == std::string::npos) end = manifest.size();
        std::string line = manifest.substr(at, end - at);
        at = end + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line_number == 0) {
            if (line != "x2w-manifest 2") return "manifest version";
            continue;
        }
        if (line_number == 1) {
            if (line != "stage cs") return "not a compute manifest";
            continue;
        }
        unsigned i = 0, x = 0, y = 0, z = 0;
        int a = 0, b = 0, used = 0;
        const auto whole = [&] { return used == static_cast<int>(line.size()); };
        if (line.empty() || (std::sscanf(line.c_str(), "group %u %u %u%n", &x, &y, &z, &used) == 3 && whole())) continue;
        if (std::sscanf(line.c_str(), "ud %u%n", &declared_uds, &used) == 1 && whole()) continue;
        if (std::sscanf(line.c_str(), "udsrc %u rootconst %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uds.size())
            tr.uds.push_back({a, b});
        else if (std::sscanf(line.c_str(), "srv %u buf table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.srvs.size())
            tr.srvs.push_back({a, b});
        else if (std::sscanf(line.c_str(), "uav %u table %d %d%n", &i, &a, &b, &used) == 3 && whole() && i == tr.uavs.size())
            tr.uavs.push_back({a, b});
        else return line.rfind("srv", 0) == 0 ? "a resource that is not a raw table buffer"
                  : line.rfind("uav", 0) == 0 ? "an output that is not a raw table buffer"
                  : line.rfind("udsrc", 0) == 0 ? "an entry value that is not a root constant"
                  : line.rfind("smp", 0) == 0 ? "a sampler" : "an unknown manifest line";
    }
    // shortcut: one constants parameter and one table with both ranges; lift when a program without entry values,
    // inputs or outputs is requested.
    if (tr.uds.empty() || tr.uds.size() != declared_uds || tr.uds.size() > 32 || tr.srvs.empty() || tr.srvs.size() > 32 ||
        tr.uavs.empty() || tr.uavs.size() > 8)
        return "entry value, input or output count";
    x2w_startup::ComPtr<ID3DBlob> code;
    if (!cache_path(path, L"", tr.key, L".cs.cso") || FAILED(D3DReadFileToBlob(path, &code))) return "no compiled program";
    if (FAILED(x2w_startup::create_translated_root(device_of(the_device), static_cast<UINT>(tr.uds.size()),
            static_cast<UINT>(tr.srvs.size()), {}, &tr.root, static_cast<UINT>(tr.uavs.size()))))
        return "root signature creation failed";
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = tr.root.Get();
    desc.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    if (FAILED(device_of(the_device)->CreateComputePipelineState(&desc, IID_PPV_ARGS(&tr.pipeline))))
        return "pipeline creation failed";
    tr.state = 1;
    log("x2w-draw compute pipeline %016llx translated: %zu entry values, %zu inputs, %zu outputs", tr.key, tr.uds.size(),
        tr.srvs.size(), tr.uavs.size());
    return nullptr;
}

// Both callers have an open list and a slot 67 request. The indirect caller has also validated the
// signature and placed argument range. All resource checks precede the first recorded state change.
const char* record_translated_dispatch(Wrapper* self, const TranslatedDispatch& dispatch)
{
    auto* list = recording(self);
    Pass& pass = pass_of(self);
    const u64 key = pass.pipeline->own().compute_request_key;
    const auto [x, y, z] = dispatch.groups;
    const char* reason = nullptr;
    LONG* counter = &compute_dispatches_refused;
    struct View { ID3D12Resource* resource; u64 offset, bytes; bool transition; } views[40]{};
    ComputeTranslation* tr = nullptr;
    const auto type = list->GetType();
    AcquireSRWLockExclusive(&draw_lock);
    if (type != D3D12_COMMAND_LIST_TYPE_DIRECT && type != D3D12_COMMAND_LIST_TYPE_COMPUTE) reason = "the list type has no Dispatch";
    else if (x > x2w_startup::max_dispatch_groups || y > x2w_startup::max_dispatch_groups || z > x2w_startup::max_dispatch_groups)
        reason = "a group count is over the limit";
    else if (pass.compute_root_wrapper != pass.pipeline->own().compute_request_root)
        reason = "the compute root set is not the one the pipeline was created with";
    else if (!ensure_draw_heap(self->own().list_device)) reason = "no descriptor heap on this device";
    else {
        tr = &compute_translations()[key];
        tr->key = key;
        if (!tr->state || (tr->retry_manifest_at && GetTickCount64() >= tr->retry_manifest_at)) {
            if (const char* why = load_compute_translation(*tr))
                log("x2w-draw compute pipeline %016llx not translated: %s", key, why);
        }
        if (tr->state != 1) {
            reason = "no translation";
            counter = &compute_dispatches_untranslated;
        }
    }
    const std::size_t inputs = reason ? 0 : tr->srvs.size(), count = reason ? 0 : inputs + tr->uavs.size();
    for (std::size_t i = 0; i < count && !reason; ++i) {
        const bool output = i >= inputs;
        const ComputeTranslation::Slot slot = output ? tr->uavs[i - inputs] : tr->srvs[i];
        const u64 entry = slot.a >= 0 && slot.b >= 0 ? pass.compute_tables.table(slot.a).record(static_cast<u64>(slot.b), 16) : 0;
        std::uint32_t words[4];
        if (!entry || !x2w_local::readable(reinterpret_cast<const void*>(entry), sizeof(words))) {
            reason = "a table descriptor is not bound";
            break;
        }
        std::memcpy(words, reinterpret_cast<const void*>(entry), sizeof(words));
        if (!output) {
            const IndirectBufferView view = indirect_table_buffer(self, words);
            if (!view.resource || view.offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT) reason = "an input is not a complete GPU-backed buffer range";
            views[i] = {view.resource, view.offset, view.bytes, !view.upload};
            continue;
        }
        // An output is written by the GPU: only a placed buffer that aliases the title's memory shows the result
        // to the title. shortcut: view tokens and copied placements are refused, add them with their first user.
        const u64 address = words[0] | (u64{words[1] & 0xffff} << 32);
        const UINT stride = (words[1] >> 16) & 0x3fff;
        const u64 bytes = u64{stride ? stride : 1} * words[2];
        const auto placed = words[0] == view_token_magic || !bytes || bytes % 4 || bytes > (16u << 20)
            ? PlacedVertexView{} : placed_buffer_range(self->own().list_device, address, bytes);
        const ResourceInfo* info = placed.alias ? resource_info(placed.owner) : nullptr;
        const u64 offset = placed.alias ? address - reinterpret_cast<u64>(placed.owner->own().title_memory) : 0;
        if (!info || info->initial_state != D3D12_RESOURCE_STATE_COMMON || offset % D3D12_RAW_UAV_SRV_BYTE_ALIGNMENT ||
            !(placed.alias->GetDesc().Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS))
            reason = "an output is not an aliased placed buffer range with unordered access";
        for (std::size_t input = 0; input < inputs && !reason; ++input)
            if (views[input].resource == placed.alias) reason = "an output buffer is also an input";
        views[i] = {placed.alias, offset, bytes, true};
    }
    if (dispatch.argument)
        for (std::size_t i = 0; i < count && !reason; ++i)
            if (views[i].resource == dispatch.argument) reason = "the indirect argument buffer is also a bound shader resource";
    if (reason) {
        ReleaseSRWLockExclusive(&draw_lock);
        InterlockedIncrement(counter);
        // Dispatches that arrive before the offline translation must not use up the lines for bind refusals.
        static LONG untranslated, refusals;
        if (!dispatch.signature && first_calls(counter == &compute_dispatches_untranslated ? untranslated : refusals, 16))
            log("x2w-draw compute %016llx Dispatch(%u, %u, %u) refused: %s", key, x, y, z, reason);
        return reason;
    }
    const UINT n = static_cast<UINT>(count);
    UINT first = static_cast<UINT>(InterlockedExchangeAdd(&draw_heap_cursor, static_cast<LONG>(n))) % draw_heap_size;
    if (first + n > draw_heap_size) {
        first = 0;
        InterlockedExchange(&draw_heap_cursor, static_cast<LONG>(n));
    }
    D3D12_CPU_DESCRIPTOR_HANDLE dest{draw_heap->GetCPUDescriptorHandleForHeapStart().ptr + std::size_t{first} * draw_heap_increment};
    for (std::size_t i = 0; i < count; ++i, dest.ptr += draw_heap_increment) {
        const View& view = views[i];
        if (i < inputs) {
            if (view.transition) require_state(self, view.resource, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            buffer_view(dest, view.resource, view.offset, static_cast<std::size_t>(view.bytes));
            continue;
        }
        require_state(self, view.resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
        desc.Format = DXGI_FORMAT_R32_TYPELESS;
        desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        desc.Buffer.FirstElement = view.offset / 4;
        desc.Buffer.NumElements = static_cast<UINT>(view.bytes / 4);
        desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        device_of(the_device)->CreateUnorderedAccessView(view.resource, nullptr, &desc, dest);
    }
    std::uint32_t values[32] = {};
    for (std::size_t i = 0; i < tr->uds.size(); ++i) values[i] = pass.compute_constants.word(tr->uds[i].a, tr->uds[i].b);
    if (!pass.heap_set) {
        x2w_startup::bind_translated_heaps(list, draw_heap, nullptr);
        pass.heap_set = true;
        pass.bound_sampler_heap = nullptr;
    }
    list->SetPipelineState(tr->pipeline.Get());
    list->SetComputeRootSignature(tr->root.Get());
    list->SetComputeRoot32BitConstants(0, static_cast<UINT>(tr->uds.size()), values, 0);
    list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{draw_heap->GetGPUDescriptorHandleForHeapStart().ptr + u64{first} * draw_heap_increment});
    if (dispatch.signature) {
        require_state(self, dispatch.argument, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
        list->ExecuteIndirect(dispatch.signature, 1, dispatch.argument, dispatch.offset, nullptr, 0);
    } else {
        list->Dispatch(x, y, z);
    }
    for (std::size_t i = inputs; i < count; ++i) {  // a later dispatch on this list reads what this one wrote
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = views[i].resource;
        list->ResourceBarrier(1, &barrier);
    }
    // The translated root replaced the desktop compute root of the title and its arguments.
    if (pass.desktop_compute_root) {
        list->SetComputeRootSignature(pass.desktop_compute_root.Get());
        for (UINT parameter = 0; parameter < pass.desktop_compute_upload_srvs.size(); ++parameter)
            if (const auto address = pass.desktop_compute_upload_srvs[parameter]) list->SetComputeRootShaderResourceView(parameter, address);
    }
    ReleaseSRWLockExclusive(&draw_lock);
    check_device("translated dispatch", dispatch.signature ? 61 : 14);
    InterlockedIncrement(&compute_dispatches_translated);
    static LONG logged;
    if (!dispatch.signature && first_calls(logged, 8))
        log("x2w-draw compute %016llx Dispatch(%u, %u, %u) translated: entry [%x %x %x %x], %zu inputs, %zu outputs", key, x, y, z,
            values[0], values[1], values[2], values[3], inputs, count - inputs);
    return nullptr;
}
