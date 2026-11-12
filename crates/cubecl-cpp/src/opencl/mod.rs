//! `OpenCL` C 3.0 (`-cl-std=CL3.0`): the shared C++ emission, with C where the shared code is C++
//! (casts, brace initialisers, lambdas, templates), `OpenCL` address spaces and built-ins, C11
//! atomics, `cl_khr_subgroup_*` plane operations, and cooperative matrices on Intel's
//! `cl_intel_subgroup_matrix_multiply_accumulate` (DPAS).
//!
//! Some lowerings are GNU statement expressions (`({ ... })`, a value computed by statements): a
//! Clang extension, which every `OpenCL` C compiler in use is built on (Intel's, `PoCL`, rusticl,
//! `ROCm`'s).

pub mod atomic;
pub mod binary;
pub mod branch;
pub mod builtin;
pub mod compiler;
pub mod dialect;
pub mod mma;
pub mod plane;
pub mod signature;
pub mod ty;
pub mod unary;
pub mod vector;

pub use compiler::{OpenClCompilationOptions, OpenClCompiler, OpenClOptions};

use dialect::{call, opencl_op, opencl_op_with_out};
