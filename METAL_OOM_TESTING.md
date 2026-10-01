# Metal: out-of-memory test

When the GPU runs out of memory, CubeCL used to panic on its device thread. It now reports a
normal error, and the buffer that could not be allocated says why when it is read. Please
check this on the Mac, on both Metal paths:

- the native Metal runtime (`cubecl-metal`);
- wgpu on Metal (`cubecl-wgpu` with the `msl` feature), which is what Burn uses on a Mac.

The Metal part of this change has never been compiled, since it was written on Linux. A
build error is a useful result too.

## Setup

In the CubeCL checkout from last time:

```sh
git fetch origin
git checkout feat/sync-point-errors-metal-test
git pull
```

Or from scratch:

```sh
git clone https://github.com/tracel-ai/cubecl && cd cubecl
git checkout feat/sync-point-errors-metal-test
```

## 1. The out-of-memory test

```sh
cargo test -p cubecl-metal --lib -- out_of_memory > metal-oom.txt 2>&1
cargo test -p cubecl-wgpu --features msl --lib -- out_of_memory > wgpu-msl-oom.txt 2>&1
```

Each runs one test, `out_of_memory::an_allocation_too_large_for_the_device_fails_at_the_read`.
It asks for a 1 TB buffer, then checks three things:

- reading that buffer fails with an error that mentions the size (`1099511627776`);
- the error does not say the device is poisoned;
- a small allocation right after it still works.

We expect it to pass on both. If it fails, the output names the failed step.

## 2. The full test suites

These check that nothing else broke:

```sh
cargo test -p cubecl-metal > cubecl-metal.txt 2>&1
cargo test -p cubecl-wgpu --features msl > cubecl-wgpu-msl.txt 2>&1
```

What we expect:

- `cubecl-metal`: everything passes except the same 4 tests as last time, which also fail on
  CubeCL's `main` branch:
  - `tests::test_create_empty_tensor`
  - `tests::test_read_empty_tensor`
  - `tests::test_split_window_closes_on_its_own_stream`
  - `tests_launch_errors::oversized_shared_memory_is_a_resource_limit_error`
- `cubecl-wgpu --features msl`: everything passes.

Any other failure is worth pointing out.

**Please send:** `metal-oom.txt`, `wgpu-msl-oom.txt`, `cubecl-metal.txt` and
`cubecl-wgpu-msl.txt`.
