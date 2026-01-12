# Performance Hacking

## Tracing

Most of the crates support the "tracing" feature, which enables the [tracing](https://crates.io/crates/tracing) crate.

When the feature is enabled, some critical path code is instrumented with tracing spans:

```rust
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(self, size))
)]
fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
    ...
}
```

### Note: *always* gate tracing behind the `tracing` feature.


## Tracing A Test

The [test-log](https://crates.io/crates/test-log) crate is threaded through many tests as a replacement for `#[test]`.

This crate provides test-time instrumentation to setup log and trace output.

Actually getting this (extremely verbose) output requires a bit of setup:
- the `tracing` feature must be enabled.
- the `test-log` output initialization must be enabled via the `test_log/default` feature.
- a sufficiently broad `RUST_LOG` environment variable must be set (this is an entire config language).
- the `--nocapture` flag must be passed to `cargo test`.

An example:
```terminaloutput
$ RUST_LOG=trace cargo test -p cubecl-cuda \
  --features tracing,test-log/default \
  tests::identity::f16_ty::test_large -- --nocapture
    Finished `test` profile [unoptimized] target(s) in 17.84s
     Running unittests src/lib.rs (target/debug/deps/cubecl_cuda-42be3205312238f1)

running 1 test
2026-01-08T23:56:09.960606Z  INFO cubecl_cuda::compute::server: Peer data transfer not available for device 0
2026-01-08T23:56:09.977977Z TRACE launch_inner{count=(4, 64, 1) mode=Unchecked stream_id=StreamId { value: 0 } kernel.name=cubecl_std::tensor::identity::identity_kernel::IdentityKernel<cubecl_cuda::runtime::CudaRuntime> kernel.id=(
    CubeDim {
        x: 16,
        y: 16,
        z: 1,
    },
    Scalar (
        Float (
            F16,
        ),
    ),
    Scalar,
    TensorCompilationArg {
        inplace: None,
        line_size: 16,
    },
)}: cubecl_cuda::compute::context: Compiling kernel
test tests::identity::f16_ty::test_large ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 641 filtered out; finished in 0.48s
```