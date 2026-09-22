//! The LLVM module and target machine the pliron conversion hands its output to: parsed,
//! finalized, optimized and emitted here, each handle owned and disposed on drop.

use llvm_sys::{
    core::{
        LLVMAddAttributeAtIndex, LLVMContextCreate, LLVMContextDispose,
        LLVMCreateMemoryBufferWithMemoryRangeCopy, LLVMCreateStringAttribute,
        LLVMDisposeMemoryBuffer, LLVMDisposeMessage, LLVMDisposeModule, LLVMGetBufferSize,
        LLVMGetBufferStart, LLVMGetNamedFunction, LLVMPrintModuleToString, LLVMSetTarget,
    },
    error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage},
    ir_reader::LLVMParseIRInContext2,
    prelude::{LLVMContextRef, LLVMModuleRef, LLVMValueRef},
    target::{LLVMDisposeTargetData, LLVMSetModuleDataLayout},
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetDataLayout,
        LLVMCreateTargetMachine, LLVMDisposeTargetMachine, LLVMGetTargetFromTriple, LLVMRelocMode,
        LLVMTargetMachineEmitToMemoryBuffer, LLVMTargetMachineRef,
    },
    transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    },
};
use std::ffi::{CStr, CString};

/// An LLVM module parsed from textual IR into a context of its own.
pub(crate) struct LlvmModule {
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
}

impl LlvmModule {
    pub(crate) fn parse(ir: &str) -> Result<Self, String> {
        // SAFETY: the context is fresh, and `LLVMParseIRInContext2` takes ownership of the
        // buffer, including on failure.
        unsafe {
            let ctx = LLVMContextCreate();
            let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
                ir.as_ptr() as *const _,
                ir.len(),
                c"kernel".as_ptr(),
            );
            let mut module = std::ptr::null_mut();
            let mut error = std::ptr::null_mut();
            if LLVMParseIRInContext2(ctx, buffer, &mut module, &mut error) != 0 {
                LLVMContextDispose(ctx);
                return Err(take_message(error));
            }
            Ok(Self { ctx, module })
        }
    }

    #[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
    pub(crate) fn context(&self) -> LLVMContextRef {
        self.ctx
    }

    #[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
    pub(crate) fn raw(&self) -> LLVMModuleRef {
        self.module
    }

    #[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
    pub(crate) fn set_triple(&self, triple: &CStr) {
        // SAFETY: the module is live for `self`'s lifetime.
        unsafe { LLVMSetTarget(self.module, triple.as_ptr()) }
    }

    #[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
    /// The function `name` defines, which a kernel's entry point must be.
    pub(crate) fn entry_point(&self, name: &str) -> Result<LLVMValueRef, String> {
        let c_name =
            CString::new(name).map_err(|_| format!("kernel name '{name}' contains a NUL"))?;
        // SAFETY: the module is live for `self`'s lifetime.
        let func = unsafe { LLVMGetNamedFunction(self.module, c_name.as_ptr()) };
        if func.is_null() {
            return Err(format!("entry point '{name}' is not defined in the module"));
        }
        Ok(func)
    }

    #[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
    /// Adds the string attributes `key=value` to `func`.
    ///
    /// # Safety
    /// `func` must be a function of this module.
    pub(crate) unsafe fn add_function_attributes(
        &self,
        func: LLVMValueRef,
        attributes: &[(&str, &str)],
    ) {
        for (key, value) in attributes {
            // SAFETY: `func` belongs to this module's context, and both strings are read for
            // the lengths given.
            unsafe {
                let attribute = LLVMCreateStringAttribute(
                    self.ctx,
                    key.as_ptr() as *const _,
                    key.len() as u32,
                    value.as_ptr() as *const _,
                    value.len() as u32,
                );
                LLVMAddAttributeAtIndex(func, llvm_sys::LLVMAttributeFunctionIndex, attribute);
            }
        }
    }

    /// Runs the pass pipeline `pipeline`, with `machine`'s cost model when there is one.
    pub(crate) fn run_passes(
        &self,
        pipeline: &CStr,
        machine: Option<&TargetMachine>,
    ) -> Result<(), String> {
        let tm = machine.map_or(std::ptr::null_mut(), |machine| machine.0);
        // SAFETY: the module and the target machine are live for the call.
        unsafe {
            let options = LLVMCreatePassBuilderOptions();
            let error = LLVMRunPasses(self.module, pipeline.as_ptr(), tm, options);
            LLVMDisposePassBuilderOptions(options);
            if error.is_null() {
                return Ok(());
            }
            let c_msg = LLVMGetErrorMessage(error);
            let message = CStr::from_ptr(c_msg).to_string_lossy().into_owned();
            LLVMDisposeErrorMessage(c_msg);
            Err(message)
        }
    }

    pub(crate) fn print(&self) -> String {
        // SAFETY: the module is live, and the printed string is ours to free.
        unsafe { take_message(LLVMPrintModuleToString(self.module)) }
    }
}

impl Drop for LlvmModule {
    fn drop(&mut self) {
        // SAFETY: both handles are owned by `self`, the module before its context.
        unsafe {
            LLVMDisposeModule(self.module);
            LLVMContextDispose(self.ctx);
        }
    }
}

#[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
/// An LLVM target machine at the aggressive optimization level.
pub(crate) struct TargetMachine(LLVMTargetMachineRef);

#[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
impl TargetMachine {
    /// The target must have been initialized.
    pub(crate) fn new(
        triple: &CStr,
        cpu: &str,
        features: &CStr,
        reloc: LLVMRelocMode,
    ) -> Result<Self, String> {
        let c_cpu = CString::new(cpu).map_err(|_| format!("arch '{cpu}' contains a NUL"))?;
        // SAFETY: every string outlives the calls, and a failed lookup hands back a message
        // we own.
        unsafe {
            let mut target = std::ptr::null_mut();
            let mut error = std::ptr::null_mut();
            if LLVMGetTargetFromTriple(triple.as_ptr(), &mut target, &mut error) != 0 {
                return Err(take_message(error));
            }
            let tm = LLVMCreateTargetMachine(
                target,
                triple.as_ptr(),
                c_cpu.as_ptr(),
                features.as_ptr(),
                LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
                reloc,
                LLVMCodeModel::LLVMCodeModelDefault,
            );
            if tm.is_null() {
                return Err(format!("no target machine for '{cpu}'"));
            }
            Ok(Self(tm))
        }
    }

    pub(crate) fn set_data_layout(&self, module: &LlvmModule) {
        // SAFETY: both are live, and the module copies the layout it is given.
        unsafe {
            let layout = LLVMCreateTargetDataLayout(self.0);
            LLVMSetModuleDataLayout(module.module, layout);
            LLVMDisposeTargetData(layout);
        }
    }

    /// Emission changes the module, so a module emits once.
    pub(crate) fn emit(
        &self,
        module: LlvmModule,
        kind: LLVMCodeGenFileType,
    ) -> Result<Vec<u8>, String> {
        // SAFETY: both are live, and the buffer handed back is ours to read and free.
        unsafe {
            let mut buffer = std::ptr::null_mut();
            let mut error = std::ptr::null_mut();
            if LLVMTargetMachineEmitToMemoryBuffer(
                self.0,
                module.module,
                kind,
                &mut error,
                &mut buffer,
            ) != 0
            {
                return Err(take_message(error));
            }
            let start = LLVMGetBufferStart(buffer) as *const u8;
            let len = LLVMGetBufferSize(buffer);
            let bytes = std::slice::from_raw_parts(start, len).to_vec();
            LLVMDisposeMemoryBuffer(buffer);
            Ok(bytes)
        }
    }
}

impl Drop for TargetMachine {
    fn drop(&mut self) {
        // SAFETY: the target machine is owned by `self`.
        unsafe { LLVMDisposeTargetMachine(self.0) }
    }
}

/// # Safety
/// `message` must be a NUL-terminated string LLVM allocated for the caller to dispose.
unsafe fn take_message(message: *mut std::ffi::c_char) -> String {
    // SAFETY: the caller's contract.
    unsafe {
        let owned = CStr::from_ptr(message).to_string_lossy().into_owned();
        LLVMDisposeMessage(message);
        owned
    }
}
