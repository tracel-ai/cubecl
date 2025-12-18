# Tracing Example

## tracing/max_level_off

Enabling `tracing/max_level_off` disables all tracing at compile time
for non-`--release` builds.

### No runtime subscriber

- No tracing subscriber is set up.
- Tracing is disabled at compile time.

```terminaloutput
$ cargo run -p tracing_example --features cuda,tracing/max_level_off
Args { tracing: None }
```

### No RUST_LOG

- Enabling console subscriber.
- The INFO log is emitted because console logging was enabled by setting up the subscriber.
- There is a warning (thrown by the tracing subscriber setup) because the subscriber uses a default filter of `INFO`.

```terminaloutput
$ cargo run -p tracing_example --features cuda,tracing/max_level_off -- --tracing console
Args { tracing: Some(Console) }
warning: some trace filter directives would enable traces that are disabled statically
 | `info` would enable the INFO level for all targets
 = note: the static max level is `off`
 = help: to enable logging, remove the `max_level_off` feature from the `tracing` crate
2025-12-18T06:22:21.376807Z  INFO cubecl_cuda::compute::server: Peer data transfer not available for device 0
```

### tracing/max_level_trace

### No runtime subscriber

- No tracing subscriber is set up.
- Tracing is enabled at compile time.

```terminaloutput
$ cargo run -p tracing_example --features cuda,tracing/max_level_trace
Args { tracing: None }
```

### No RUST_LOG

- Enabling console subscriber.
- The INFO log is emitted because console logging was enabled by setting up the subscriber.
- No `RUST_LOG`, so the default filter of `INFO` strips out all the `cubecl` trace-level events.

```terminaloutput
$ cargo run -p tracing_example --features cuda,tracing/max_level_trace -- --tracing console
Args { tracing: Some(Console) }
2025-12-18T06:28:09.891230Z  INFO cubecl_cuda::compute::server: Peer data transfer not available for device 0
```

### RUST_LOG=trace

- Enabling console subscriber.
- The INFO log is emitted because console logging was enabled by setting up the subscriber.
- The `RUST_LOG` enables everything at `TRACE` level or below.

```terminaloutput
crutcher@HeatLamp:~/git/cubecl$ RUST_LOG=trace cargo run -p tracing_example --features cuda,tracing/max_level_trace -- --tracing console
Args { tracing: Some(Console) }
2025-12-18T06:30:42.979458Z TRACE launch: tracing_example: enter
2025-12-18T06:30:43.267396Z  INFO launch: cubecl_cuda::compute::server: Peer data transfer not available for device 0
2025-12-18T06:30:43.267663Z TRACE launch:reserve{size=512}: cubecl_cuda::compute::command: enter
2025-12-18T06:30:43.267688Z TRACE launch:reserve{size=512}:reserve{size=512}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:30:43.267710Z TRACE launch:reserve{size=512}:reserve{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:30:43.267731Z TRACE launch:reserve{size=512}:reserve{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:30:43.267760Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:30:43.267784Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:30:43.267805Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}:alloc{size=512}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:30:43.275268Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}:alloc{size=512}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:30:43.275360Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}:alloc{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_pool::memory_page: enter
2025-12-18T06:30:43.275390Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}:alloc{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_pool::memory_page: exit
2025-12-18T06:30:43.275425Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:30:43.275446Z TRACE launch:reserve{size=512}:reserve{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:30:43.275463Z TRACE launch:reserve{size=512}:reserve{size=512}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:30:43.275479Z TRACE launch:reserve{size=512}: cubecl_cuda::compute::command: exit
2025-12-18T06:30:43.275533Z TRACE launch:write_to_gpu: cubecl_cuda::compute::command: enter
2025-12-18T06:30:43.275565Z TRACE launch:write_to_gpu:write_to_gpu{shape=[16] elem_size=1}: cubecl_cuda::compute::command: enter
2025-12-18T06:30:43.275604Z TRACE launch:write_to_gpu:write_to_gpu{shape=[16] elem_size=1}: cubecl_cuda::compute::command: exit
2025-12-18T06:30:43.275651Z TRACE launch:write_to_gpu: cubecl_cuda::compute::command: exit
2025-12-18T06:30:43.275696Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}: cubecl_cuda::compute::command: enter
2025-12-18T06:30:43.275717Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}: cubecl_cuda::compute::command: enter
2025-12-18T06:30:43.275736Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:30:43.275757Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:30:43.275776Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:30:43.275802Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:30:43.275825Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:30:43.275847Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}:alloc{size=16}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:30:43.290707Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}:alloc{size=16}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:30:43.290798Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}:alloc{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_pool::memory_page: enter
2025-12-18T06:30:43.290829Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}:alloc{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_pool::memory_page: exit
2025-12-18T06:30:43.290869Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:30:43.290893Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:30:43.290914Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}:reserve{size=16}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:30:43.290943Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}:reserve_pinned{size=16 origin=None}: cubecl_cuda::compute::command: exit
2025-12-18T06:30:43.290960Z TRACE launch:reserve_cpu{size=16 marked_pinned=true origin=None}: cubecl_cuda::compute::command: exit
2025-12-18T06:30:43.290995Z TRACE launch:write_to_cpu{shape=[16] elem_size=1}: cubecl_cuda::compute::command: enter
2025-12-18T06:30:43.291032Z TRACE launch:write_to_cpu{shape=[16] elem_size=1}: cubecl_cuda::compute::command: exit
2025-12-18T06:30:43.291070Z TRACE launch:block_on: cubecl_common::future: enter
2025-12-18T06:30:43.291120Z TRACE launch:block_on: cubecl_common::future: exit
2025-12-18T06:30:43.291140Z TRACE launch: tracing_example: exit
```

### RUST_LOG=off,cubeccl_runtime=trace

- Enabling console subscriber.
- `RUST_LOG=off,cubeccl_runtime=trace` enables tracing for the `cubeccl_runtime` crate only.

```terminaloutput
$ RUST_LOG=off,cubecl_runtime=trace cargo run -p tracing_example --features cuda,tracing/max_level_trace -- --tracing console
Args { tracing: Some(Console) }
2025-12-18T06:33:06.994510Z TRACE reserve{size=512}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:33:06.994557Z TRACE reserve{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:33:06.994579Z TRACE reserve{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:33:06.994609Z TRACE reserve{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:33:06.994632Z TRACE reserve{size=512}:alloc{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:33:06.994654Z TRACE reserve{size=512}:alloc{size=512}:alloc{size=512}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:33:07.002198Z TRACE reserve{size=512}:alloc{size=512}:alloc{size=512}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:33:07.002279Z TRACE reserve{size=512}:alloc{size=512}:alloc{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_pool::memory_page: enter
2025-12-18T06:33:07.002307Z TRACE reserve{size=512}:alloc{size=512}:alloc{size=512}:try_reserve{size=512}: cubecl_runtime::memory_management::memory_pool::memory_page: exit
2025-12-18T06:33:07.002342Z TRACE reserve{size=512}:alloc{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:33:07.002361Z TRACE reserve{size=512}:alloc{size=512}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:33:07.002378Z TRACE reserve{size=512}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:33:07.002526Z TRACE reserve{size=16}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:33:07.002550Z TRACE reserve{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:33:07.002565Z TRACE reserve{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:33:07.002589Z TRACE reserve{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_manage: enter
2025-12-18T06:33:07.002610Z TRACE reserve{size=16}:alloc{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:33:07.002629Z TRACE reserve{size=16}:alloc{size=16}:alloc{size=16}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: enter
2025-12-18T06:33:07.018132Z TRACE reserve{size=16}:alloc{size=16}:alloc{size=16}:storage.alloc: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:33:07.018218Z TRACE reserve{size=16}:alloc{size=16}:alloc{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_pool::memory_page: enter
2025-12-18T06:33:07.018247Z TRACE reserve{size=16}:alloc{size=16}:alloc{size=16}:try_reserve{size=16}: cubecl_runtime::memory_management::memory_pool::memory_page: exit
2025-12-18T06:33:07.018286Z TRACE reserve{size=16}:alloc{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_pool::sliced_pool: exit
2025-12-18T06:33:07.018306Z TRACE reserve{size=16}:alloc{size=16}: cubecl_runtime::memory_management::memory_manage: exit
2025-12-18T06:33:07.018324Z TRACE reserve{size=16}: cubecl_runtime::memory_management::memory_manage: exit
```