//! CPU target.

pub mod abi;
pub mod entrypoint;
pub mod f16_evaluation;
pub mod jit;
pub mod ordered_atomic;
pub mod shared_memory;
pub mod synchronization;
pub mod to_llvm;

/// Host CPU name and feature string from LLVM.
pub fn host_cpu_target() -> (String, String) {
    crate::cpu::jit::engine::init_native();
    unsafe {
        let cpu_ptr = llvm_sys::target_machine::LLVMGetHostCPUName();
        let features_ptr = llvm_sys::target_machine::LLVMGetHostCPUFeatures();
        let cpu = std::ffi::CStr::from_ptr(cpu_ptr)
            .to_string_lossy()
            .into_owned();
        let features = std::ffi::CStr::from_ptr(features_ptr)
            .to_string_lossy()
            .into_owned();
        llvm_sys::core::LLVMDisposeMessage(cpu_ptr);
        llvm_sys::core::LLVMDisposeMessage(features_ptr);
        (cpu, features)
    }
}
