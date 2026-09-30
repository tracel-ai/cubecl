# Metal follow-up: a real GPU fault on the native runtime

Thanks for the first round of Metal testing. Two things came out of it:

- `cubecl-metal` did not compile without the three small fixes you made. They are on this
  branch now, so it should build as is.
- Neither run could make the GPU actually fault: Burn's Metal path goes through wgpu, which
  blocks the out-of-bounds write. This branch adds a test that forces a real GPU fault on the
  native Metal runtime instead.

Please run the two parts below on the same Mac and send back the two output files.

## Setup

```sh
git clone https://github.com/tracel-ai/cubecl && cd cubecl
git checkout feat/sync-point-errors-metal-test
```

## 1. The GPU fault test

```sh
cargo test -p cubecl-metal --test device_fault -- --nocapture > metal-device-fault.txt 2>&1
```

The test launches a kernel that writes a gigabyte past the end of its buffer, which should
make the GPU raise an address fault. It then reads that buffer, syncs, launches more work, and
does a sync and a read from another thread.

What we expect:

- The test passes: the read of the faulted buffer and the sync each return an error, and
  nothing panics.
- For each step, it prints a line like `sync: device poisoned = false` followed by the error
  message. These lines are what we most want to see: whether the fault reaches the later
  steps and the other thread, and how it is reported.

If the test fails instead, the output shows which step, which is just as useful.

## 2. The full native Metal test suite

```sh
cargo test -p cubecl-metal > cubecl-metal.txt 2>&1
```

This is the same suite as last time, now without any manual fix. Last time 4 tests failed,
and they fail on CubeCL's `main` branch too, so we expect the same 4:

- `tests::test_create_empty_tensor`
- `tests::test_read_empty_tensor`
- `tests::test_split_window_closes_on_its_own_stream`
- `tests_launch_errors::oversized_shared_memory_is_a_resource_limit_error`

Any other failure, or a build error, is worth pointing out.

**Please send:** `metal-device-fault.txt` and `cubecl-metal.txt`.
