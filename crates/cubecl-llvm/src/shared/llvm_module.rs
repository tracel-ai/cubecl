//! The LLVM module and target machine the pliron conversion hands its output to: parsed,
//! finalized, optimized and emitted here, each handle owned and disposed on drop.
//!
//! The CPU target parses, optimizes and prints a module; finalizing an entry point and
//! building a target machine are the GPU targets' alone, and are compiled only with them.

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
use llvm_sys::{
    LLVMAtomicOrdering, LLVMAttributeFunctionIndex, LLVMTypeKind,
    core::{
        LLVMAddAttributeAtIndex, LLVMCountParams, LLVMCreateEnumAttribute,
        LLVMCreateStringAttribute, LLVMGetBufferSize, LLVMGetBufferStart,
        LLVMGetEnumAttributeKindForName, LLVMGetFirstBasicBlock, LLVMGetFirstInstruction,
        LLVMGetNamedFunction, LLVMGetNextBasicBlock, LLVMGetNextInstruction, LLVMGetOrdering,
        LLVMGetParam, LLVMGetTypeKind, LLVMIsALoadInst, LLVMSetFunctionCallConv, LLVMSetTarget,
        LLVMTypeOf,
    },
    prelude::LLVMValueRef,
    target::{LLVMDisposeTargetData, LLVMSetModuleDataLayout},
    target_machine::{
        LLVMCodeGenFileType, LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetDataLayout,
        LLVMCreateTargetMachine, LLVMGetTargetFromTriple, LLVMRelocMode,
        LLVMTargetMachineEmitToMemoryBuffer,
    },
};
use llvm_sys::{
    core::{
        LLVMContextCreate, LLVMContextDispose, LLVMCreateMemoryBufferWithMemoryRangeCopy,
        LLVMDisposeMemoryBuffer, LLVMDisposeMessage, LLVMDisposeModule, LLVMPrintModuleToString,
    },
    error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage},
    ir_reader::LLVMParseIRInContext2,
    prelude::{LLVMContextRef, LLVMModuleRef},
    target_machine::{LLVMDisposeTargetMachine, LLVMTargetMachineRef},
    transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    },
};
use std::ffi::CStr;
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
use std::{ffi::CString, marker::PhantomData};

/// An LLVM module parsed from textual IR into a context of its own.
pub(crate) struct LlvmModule {
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
}

impl LlvmModule {
    /// Parses the textual IR `ir` into a context of its own.
    ///
    /// # Errors
    /// The message LLVM's parser gives, when `ir` is not valid LLVM IR.
    pub(crate) fn new(ir: &str) -> Result<Self, String> {
        // SAFETY: the context is fresh, and `LLVMParseIRInContext2` does not consume the
        // buffer, so it is disposed here once parsing is done, on either path.
        unsafe {
            let ctx = LLVMContextCreate();
            let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
                ir.as_ptr() as *const _,
                ir.len(),
                c"kernel".as_ptr(),
            );
            let mut module = std::ptr::null_mut();
            let mut error = std::ptr::null_mut();
            let failed = LLVMParseIRInContext2(ctx, buffer, &mut module, &mut error) != 0;
            LLVMDisposeMemoryBuffer(buffer);
            if failed {
                LLVMContextDispose(ctx);
                return Err(take_message(error));
            }
            Ok(Self { ctx, module })
        }
    }

    #[cfg(any(feature = "amdgpu", feature = "nvptx"))]
    pub(crate) fn raw(&self) -> LLVMModuleRef {
        self.module
    }

    #[cfg(any(feature = "amdgpu", feature = "nvptx"))]
    pub(crate) fn set_triple(&self, triple: &CStr) {
        // SAFETY: the module is live for `self`'s lifetime.
        unsafe { LLVMSetTarget(self.module, triple.as_ptr()) }
    }

    /// The function `name` defines, which a kernel's entry point must be.
    ///
    /// # Errors
    /// When the module defines no such function, or `name` contains a NUL.
    #[cfg(any(feature = "amdgpu", feature = "nvptx"))]
    pub(crate) fn entry_point(&self, name: &str) -> Result<EntryFunction<'_>, String> {
        let c_name =
            CString::new(name).map_err(|_| format!("kernel name '{name}' contains a NUL"))?;
        // SAFETY: the module is live for `self`'s lifetime.
        let func = unsafe { LLVMGetNamedFunction(self.module, c_name.as_ptr()) };
        if func.is_null() {
            return Err(format!("entry point '{name}' is not defined in the module"));
        }
        Ok(EntryFunction { module: self, func })
    }

    /// Adds the module flag `key = value`, which a module linked into this one must agree
    /// with.
    #[cfg(feature = "amdgpu")]
    pub(crate) fn add_module_flag(&self, key: &str, value: u32) {
        use llvm_sys::LLVMModuleFlagBehavior;
        use llvm_sys::core::{
            LLVMAddModuleFlag, LLVMConstInt, LLVMInt32TypeInContext, LLVMValueAsMetadata,
        };

        // SAFETY: the module and its context are live, and `key` is read for the length given.
        unsafe {
            let value = LLVMConstInt(LLVMInt32TypeInContext(self.ctx), value as u64, 0);
            LLVMAddModuleFlag(
                self.module,
                LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorError,
                key.as_ptr() as *const _,
                key.len(),
                LLVMValueAsMetadata(value),
            );
        }
    }

    /// The metadata kind `name` names in this module's context.
    #[cfg(feature = "amdgpu")]
    fn metadata_kind(&self, name: &str) -> u32 {
        use llvm_sys::core::LLVMGetMDKindIDInContext;

        // SAFETY: the context is live and `name` is read for the length given.
        unsafe { LLVMGetMDKindIDInContext(self.ctx, name.as_ptr() as *const _, name.len() as u32) }
    }

    /// Runs the pass pipeline `pipeline`, with `machine`'s cost model when there is one.
    ///
    /// # Errors
    /// The message LLVM gives, when it cannot parse or run the pipeline.
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

/// A kernel's entry point in an [`LlvmModule`], and what finalizing a kernel does to it.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) struct EntryFunction<'m> {
    module: &'m LlvmModule,
    func: LLVMValueRef,
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
impl<'m> EntryFunction<'m> {
    pub(crate) fn set_calling_convention(&self, convention: u32) {
        // SAFETY: the function is live for the module's lifetime.
        unsafe { LLVMSetFunctionCallConv(self.func, convention) }
    }

    /// Adds the string attributes `key=value` to the function.
    pub(crate) fn add_attributes(&self, attributes: &[(&str, &str)]) {
        for (key, value) in attributes {
            // SAFETY: the function belongs to the module's context, and both strings are read
            // for the lengths given.
            unsafe {
                let attribute = LLVMCreateStringAttribute(
                    self.module.ctx,
                    key.as_ptr() as *const _,
                    key.len() as u32,
                    value.as_ptr() as *const _,
                    value.len() as u32,
                );
                LLVMAddAttributeAtIndex(self.func, LLVMAttributeFunctionIndex, attribute);
            }
        }
    }

    fn param_count(&self) -> u32 {
        // SAFETY: the function is live.
        unsafe { LLVMCountParams(self.func) }
    }

    /// The parameters before the last `trailing`, then the last `trailing`, each in declaration
    /// order: the entry ABI puts the buffers first and appends what the target passes after
    /// them.
    pub(crate) fn split_params(
        &self,
        trailing: u32,
    ) -> (
        impl Iterator<Item = Param> + use<>,
        impl Iterator<Item = Param> + use<>,
    ) {
        let count = self.param_count();
        let first_trailing = count.saturating_sub(trailing);
        (
            (0..first_trailing).map(Param),
            (first_trailing..count).map(Param),
        )
    }

    /// The parameter the entry ABI lowering appended last, or `None` for a function with none.
    #[cfg(feature = "nvptx")]
    pub(crate) fn last_param(&self) -> Option<Param> {
        self.split_params(1).1.next()
    }

    pub(crate) fn param_is_pointer(&self, param: Param) -> bool {
        // SAFETY: the function is live, and a `Param` names one of its parameters.
        unsafe {
            LLVMGetTypeKind(LLVMTypeOf(LLVMGetParam(self.func, param.0)))
                == LLVMTypeKind::LLVMPointerTypeKind
        }
    }

    /// Adds the enum attribute `name`, with `value`, to `param`. Returns `false` when this LLVM
    /// has no such attribute.
    #[must_use]
    pub(crate) fn add_param_attribute(&self, param: Param, name: &str, value: u64) -> bool {
        let Some(kind) = enum_attribute_kind(name) else {
            return false;
        };
        // SAFETY: the function belongs to the module's context, and a `Param` names one of its
        // parameters.
        unsafe {
            let attribute = LLVMCreateEnumAttribute(self.module.ctx, kind, value);
            LLVMAddAttributeAtIndex(self.func, param.attribute_index(), attribute);
        }
        true
    }

    /// Declares `param` a by-value block of `bytes` bytes. Returns `false` when this LLVM has no
    /// `byval` attribute.
    #[cfg(feature = "nvptx")]
    #[must_use]
    pub(crate) fn add_param_byval(&self, param: Param, bytes: u64) -> bool {
        use llvm_sys::core::{LLVMArrayType2, LLVMCreateTypeAttribute, LLVMInt8TypeInContext};

        let Some(byval) = enum_attribute_kind("byval") else {
            return false;
        };
        // SAFETY: as for `add_param_attribute`; the block type lives in the module's context.
        unsafe {
            let block = LLVMArrayType2(LLVMInt8TypeInContext(self.module.ctx), bytes);
            let attribute = LLVMCreateTypeAttribute(self.module.ctx, byval, block);
            LLVMAddAttributeAtIndex(self.func, param.attribute_index(), attribute);
        }
        true
    }

    /// Attaches the metadata `kind`, a tuple of 32-bit integers, to the function.
    #[cfg(feature = "amdgpu")]
    pub(crate) fn set_metadata(&self, kind: &str, values: &[u32]) {
        use llvm_sys::core::{
            LLVMConstInt, LLVMGlobalSetMetadata, LLVMInt32TypeInContext, LLVMMDNodeInContext2,
            LLVMValueAsMetadata,
        };

        let kind = self.module.metadata_kind(kind);
        // SAFETY: the constants and the node live in the module's context, and the function is
        // live.
        unsafe {
            let i32_ty = LLVMInt32TypeInContext(self.module.ctx);
            let mut operands: Vec<_> = values
                .iter()
                .map(|value| LLVMValueAsMetadata(LLVMConstInt(i32_ty, *value as u64, 0)))
                .collect();
            let node = LLVMMDNodeInContext2(self.module.ctx, operands.as_mut_ptr(), operands.len());
            LLVMGlobalSetMetadata(self.func, kind, node);
        }
    }

    /// Attaches the metadata `kind` to `inst`, one of this function's instructions, as an empty
    /// node: a flag that holds by being present.
    #[cfg(feature = "amdgpu")]
    pub(crate) fn set_flag_metadata(&self, inst: &Instruction<'_>, kind: &str) {
        use llvm_sys::core::{LLVMMDNodeInContext2, LLVMMetadataAsValue, LLVMSetMetadata};

        let kind = self.module.metadata_kind(kind);
        // SAFETY: the node lives in the module's context, and the instruction is live.
        unsafe {
            let empty = LLVMMDNodeInContext2(self.module.ctx, std::ptr::null_mut(), 0);
            LLVMSetMetadata(inst.inst, kind, LLVMMetadataAsValue(self.module.ctx, empty));
        }
    }

    /// The function's instructions, block by block.
    pub(crate) fn instructions(&self) -> impl Iterator<Item = Instruction<'m>> + use<'m> {
        // SAFETY: the function, its blocks and their instructions are live for the module's
        // lifetime, and the walk only reads the links between them.
        let mut block = unsafe { LLVMGetFirstBasicBlock(self.func) };
        let mut inst = if block.is_null() {
            std::ptr::null_mut()
        } else {
            unsafe { LLVMGetFirstInstruction(block) }
        };
        std::iter::from_fn(move || unsafe {
            while inst.is_null() {
                if block.is_null() {
                    return None;
                }
                block = LLVMGetNextBasicBlock(block);
                if block.is_null() {
                    return None;
                }
                inst = LLVMGetFirstInstruction(block);
            }
            let current = inst;
            inst = LLVMGetNextInstruction(inst);
            Some(Instruction {
                inst: current,
                module: PhantomData,
            })
        })
    }
}

/// An instruction of an [`EntryFunction`].
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) struct Instruction<'m> {
    inst: LLVMValueRef,
    module: PhantomData<&'m LlvmModule>,
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
impl Instruction<'_> {
    pub(crate) fn is_atomic_load(&self) -> bool {
        // SAFETY: the instruction is live.
        unsafe {
            !LLVMIsALoadInst(self.inst).is_null()
                && LLVMGetOrdering(self.inst) != LLVMAtomicOrdering::LLVMAtomicOrderingNotAtomic
        }
    }

    /// The operation of an `atomicrmw`, or `None` for any other instruction.
    #[cfg(feature = "amdgpu")]
    pub(crate) fn atomic_rmw_op(&self) -> Option<llvm_sys::LLVMAtomicRMWBinOp> {
        use llvm_sys::core::{LLVMGetAtomicRMWBinOp, LLVMIsAAtomicRMWInst};

        // SAFETY: the instruction is live, and only an `atomicrmw` is asked for its operation.
        unsafe {
            (!LLVMIsAAtomicRMWInst(self.inst).is_null()).then(|| LLVMGetAtomicRMWBinOp(self.inst))
        }
    }
}

/// One parameter of an [`EntryFunction`], which is the only thing that hands one out: an index
/// of its own would name a parameter the function may not have.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Param(u32);

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
impl Param {
    /// Its attribute index: 0 is the return value, the parameters follow.
    fn attribute_index(self) -> u32 {
        self.0 + 1
    }
}

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
fn enum_attribute_kind(name: &str) -> Option<u32> {
    // SAFETY: `name` is read for the length given.
    let kind = unsafe { LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len()) };
    (kind != 0).then_some(kind)
}

/// What a target machine is built from.
#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
pub(crate) struct TargetSpec<'a> {
    pub triple: &'a CStr,
    /// The architecture, as the target names it: `sm_60`, `gfx1201`.
    pub cpu: &'a str,
    pub features: &'a CStr,
    pub reloc: LLVMRelocMode,
}

/// An LLVM target machine at the aggressive optimization level.
// The CPU runs its passes without one, so only the GPU targets ever build it.
#[cfg_attr(not(any(feature = "amdgpu", feature = "nvptx")), allow(dead_code))]
pub(crate) struct TargetMachine(LLVMTargetMachineRef);

#[cfg(any(feature = "amdgpu", feature = "nvptx"))]
impl TargetMachine {
    /// The target `spec` names must have been initialized.
    ///
    /// # Errors
    /// When LLVM has no such target, cannot build a machine for it, or the architecture
    /// contains a NUL.
    pub(crate) fn new(spec: &TargetSpec<'_>) -> Result<Self, String> {
        let cpu =
            CString::new(spec.cpu).map_err(|_| format!("arch '{}' contains a NUL", spec.cpu))?;
        // SAFETY: every string outlives the calls, and a failed lookup hands back a message
        // we own.
        unsafe {
            let mut target = std::ptr::null_mut();
            let mut error = std::ptr::null_mut();
            if LLVMGetTargetFromTriple(spec.triple.as_ptr(), &mut target, &mut error) != 0 {
                return Err(take_message(error));
            }
            let tm = LLVMCreateTargetMachine(
                target,
                spec.triple.as_ptr(),
                cpu.as_ptr(),
                spec.features.as_ptr(),
                LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
                spec.reloc,
                LLVMCodeModel::LLVMCodeModelDefault,
            );
            if tm.is_null() {
                return Err(format!("no target machine for '{}'", spec.cpu));
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
    ///
    /// # Errors
    /// The message LLVM gives, when it cannot emit this module for this machine.
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
