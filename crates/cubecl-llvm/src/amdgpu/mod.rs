//! The AMDGPU (`amdgcn-amd-amdhsa`) target.

pub mod abi;
pub mod builtins;
pub mod codegen;
pub mod device_libs;
pub mod intrinsic;
pub mod lld;
pub mod matrix;
pub mod ocml;
pub mod plane;
pub mod plane_reduce;
pub mod printf;
pub mod shared_memory;
pub mod synchronization;
