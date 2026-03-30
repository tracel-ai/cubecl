# cubecl-hip Memory Page Violation Investigation

## High Likelihood

### 1. Default (blocking) stream creation vs NonBlocking
`src/compute/stream.rs:49-55`

CUDA explicitly creates `NonBlocking` streams. HIP uses `hipStreamCreate` with no flags (default = potentially blocking). With multiple streams, a blocking stream can serialize operations that the runtime expects to run concurrently. This can cause timing-dependent issues where a buffer is freed on one stream while another stream still references it — the synchronization model assumes non-blocking behavior.

### 2. Synchronous `hipFree` instead of async-aware deallocation
`src/compute/storage/gpu.rs:49-57`

CUDA tracks whether each allocation was made with `malloc_async` or `malloc_sync` and frees accordingly. HIP calls `hipFree` (synchronous) for everything, even though allocations are made with `hipMallocAsync`. Calling synchronous free on an async allocation can cause the free to complete before in-flight kernels on other streams finish using the memory — classic use-after-free / page violation.

### 3. No allocation kind tracking
`src/compute/storage/gpu.rs:19`

CUDA stores `(CUdeviceptr, AllocationKind)` per allocation. HIP stores only `hipDeviceptr_t`. This means HIP cannot distinguish how memory was allocated, so it can't choose the correct deallocation strategy. This compounds issue #2.

### 4. Drop queue flush ordering in kernel launch
`src/compute/command.rs:383-396`

HIP checks errors before flushing the drop queue. CUDA flushes before error checking. If a kernel fails, HIP may skip the drop queue flush entirely, leaving stale CPU buffers that get freed while GPU is still reading them.

## Medium Likelihood

### 5. No async allocation fallback
`src/compute/storage/gpu.rs:126-141`

CUDA tries `malloc_async` first, then falls back to `malloc_sync` on failure. HIP only tries `hipMallocAsync` with no fallback. If the async allocator returns a partially-valid or pool-recycled pointer, there's no safety net.

### 6. No error checking on `hipFree`
`src/compute/storage/gpu.rs:49-57`

CUDA at least logs errors from `free_sync`. HIP doesn't check the return value of `hipFree` at all. A failed free goes unnoticed, potentially corrupting the memory allocator state.

### 7. Ring buffer size hardcoded
`src/compute/storage/gpu.rs:72`

Hardcoded to `1024 * 32 = 32,768` entries. With multiple streams running concurrently, the ring buffer wraps around, and old pointer bindings get overwritten while kernels on other streams may still be using them.

### 8. Assert-based error handling in memcpy operations
`src/compute/command.rs:504,510`

HIP uses `assert_eq!` on `hipMemcpy*Async` return codes, which panics in debug but is a no-op in release builds. Memory transfer failures are silently ignored in production.

## Lower Likelihood

### 9. `memory_usage` ignores errors
`src/compute/server.rs:211-212`

HIP sets `ignore: true` for error mode on memory queries. CUDA sets `ignore: false`.

### 10. Fence event flag differences
`src/compute/fence.rs:28-43`

HIP's `hipEventDefault` behavior may differ from CUDA's `CU_EVENT_DEFAULT` on AMD hardware — particularly around event completion visibility across streams.

### 11. Pinned memory deallocation is synchronous
`src/compute/storage/cpu.rs:117`

`hipFreeHost` is called immediately on dealloc. If an async DtoH copy is still in flight on another stream referencing this pinned buffer, freeing it causes a page violation.

### 12. No file-backed allocation handling
`src/compute/command.rs` (missing vs CUDA)

CUDA converts file-backed data to a pinned intermediate buffer before GPU transfer. HIP writes directly, which could cause issues if the source data is memory-mapped or paged out.
