#![cfg(any(target_os = "linux", target_os = "windows"))]

//! With register spills refused, an autotune whose first candidate spills picks the one that
//! fits its registers. A binary of its own: the test sets the process's configuration.

type TestRuntime = cubecl_cuda::CudaRuntime;

cubecl_core::testgen_register_spills!();
