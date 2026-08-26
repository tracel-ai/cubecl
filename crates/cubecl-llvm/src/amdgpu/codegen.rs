//! Compiling the LLVM dialect to an AMD code object.

use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron_llvm::llvm_sys::core::LLVMContext;
use pliron_llvm::to_llvm_ir;
use std::ffi::{CStr, CString};
use std::sync::Once;

use crate::amdgpu::lld::link_relocatable;
use crate::amdgpu::plane_dim_for;
use crate::shared::AmdGpuModule;

/// The HSA target triple; the specific device is the `-mcpu`, not the triple.
const TRIPLE: &CStr = c"amdgcn-amd-amdhsa";

/// Code object version. v5 is what gfx1201's HSA loader accepts.
const CODE_OBJECT_VERSION: u32 = 500;

/// LLVM's calling convention number for `amdgpu_kernel`
/// (`llvm::CallingConv::AMDGPU_KERNEL`).
const AMDGPU_KERNEL_CC: u32 = 91;

/// The subtarget feature that selects wave32 on RDNA. Passed both as a function
/// attribute and to the target machine so the two can never disagree.
const WAVE32: &str = "+wavefrontsize32";

/// The pipeline run before machine-code emission. Same shape as the CPU path's,
/// but handed the target machine, which is what makes it target-aware.
const PASS_PIPELINE: &CStr = c"default<O3>";

static INIT_AMDGPU: Once = Once::new();

fn init_amdgpu() {
    INIT_AMDGPU.call_once(|| unsafe {
        llvm_sys::target::LLVMInitializeAMDGPUTargetInfo();
        llvm_sys::target::LLVMInitializeAMDGPUTarget();
        llvm_sys::target::LLVMInitializeAMDGPUTargetMC();
        llvm_sys::target::LLVMInitializeAMDGPUAsmPrinter();
    });
}

/// Subtarget feature string for `arch`, empty on the wave64 parts.
fn features_for(arch: &str) -> &'static str {
    if plane_dim_for(arch) == 32 {
        WAVE32
    } else {
        ""
    }
}

/// Lowers `module` to LLVM IR, compiles it to AMDGPU machine code, and links it.
pub fn emit_code_object(
    ctx: &Context,
    module: ModuleOp,
    entrypoint: &str,
    arch: &str,
    cube_dim: u32,
) -> Result<AmdGpuModule, String> {
    let llvm_ctx = LLVMContext::default();
    let llvm_module =
        to_llvm_ir::convert_module(ctx, &llvm_ctx, module).map_err(|err| err.to_string())?;

    let ir = finalize_ir(&llvm_module.to_string(), entrypoint, arch, cube_dim)?;
    let want_asm = std::env::var_os("CUBECL_DEBUG_PLIRON").is_some();
    let (object, asm) = compile_to_object(&ir, arch, want_asm)?;

    #[cfg(feature = "pliron-dump")]
    if let Some(dir) = crate::shared::jit::engine::ir_dump_path(entrypoint) {
        let _ = std::fs::write(dir.join("amdgpu.ll"), &ir);
        if let Some(asm) = &asm {
            let _ = std::fs::write(dir.join("amdgpu.s"), asm);
        }
    }

    let code_object = link_relocatable(&object, entrypoint)?;

    Ok(AmdGpuModule {
        code_object,
        entrypoint: entrypoint.to_string(),
        ir,
        asm,
    })
}

/// Stamps `ir` with what the AMDGPU backend keys off: the HSA triple, the
/// `amdgpu_kernel` calling convention on `entrypoint`, the subtarget attributes,
/// and the code object version.
fn finalize_ir(ir: &str, entrypoint: &str, arch: &str, cube_dim: u32) -> Result<String, String> {
    use llvm_sys::LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorError;
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMAddModuleFlag, LLVMConstInt, LLVMContextDispose,
        LLVMCreateStringAttribute, LLVMDisposeMessage, LLVMDisposeModule, LLVMGetNamedFunction,
        LLVMInt32TypeInContext, LLVMPrintModuleToString, LLVMSetFunctionCallConv, LLVMSetTarget,
        LLVMValueAsMetadata,
    };

    // Built before the module is parsed so this error path has nothing to dispose.
    let name = CString::new(entrypoint)
        .map_err(|_| format!("kernel name '{entrypoint}' contains a NUL"))?;

    let flat_work_group_size = format!("1,{cube_dim}");
    let mut attributes = vec![
        ("target-cpu", arch),
        ("amdgpu-flat-work-group-size", &flat_work_group_size),
    ];
    let features = features_for(arch);
    if !features.is_empty() {
        attributes.push(("target-features", features));
    }

    unsafe {
        let (ctx, module) = parse_ir(ir)?;

        LLVMSetTarget(module, TRIPLE.as_ptr());

        let func = LLVMGetNamedFunction(module, name.as_ptr());
        if func.is_null() {
            LLVMDisposeModule(module);
            LLVMContextDispose(ctx);
            return Err(format!(
                "entry point '{entrypoint}' is not defined in the module"
            ));
        }
        LLVMSetFunctionCallConv(func, AMDGPU_KERNEL_CC);

        for (key, value) in attributes {
            let attribute = LLVMCreateStringAttribute(
                ctx,
                key.as_ptr() as *const _,
                key.len() as u32,
                value.as_ptr() as *const _,
                value.len() as u32,
            );
            LLVMAddAttributeAtIndex(func, llvm_sys::LLVMAttributeFunctionIndex, attribute);
        }

        let version = LLVMConstInt(LLVMInt32TypeInContext(ctx), CODE_OBJECT_VERSION as u64, 0);
        let key = "amdhsa_code_object_version";
        LLVMAddModuleFlag(
            module,
            LLVMModuleFlagBehaviorError,
            key.as_ptr() as *const _,
            key.len(),
            LLVMValueAsMetadata(version),
        );

        let c_ir = LLVMPrintModuleToString(module);
        let finalized = CStr::from_ptr(c_ir).to_string_lossy().into_owned();
        LLVMDisposeMessage(c_ir);
        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        Ok(finalized)
    }
}

/// Compiles `ir` to an AMDGPU relocatable object, and to assembly alongside it
/// when `want_asm`.
fn compile_to_object(
    ir: &str,
    arch: &str,
    want_asm: bool,
) -> Result<(Vec<u8>, Option<String>), String> {
    use llvm_sys::core::{LLVMContextDispose, LLVMDisposeMessage, LLVMDisposeModule};
    use llvm_sys::target::{LLVMDisposeTargetData, LLVMSetModuleDataLayout};
    use llvm_sys::target_machine::{
        LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetDataLayout, LLVMCreateTargetMachine,
        LLVMDisposeTargetMachine, LLVMGetTargetFromTriple, LLVMRelocMode,
    };

    init_amdgpu();

    let cpu = CString::new(arch).map_err(|_| format!("arch '{arch}' contains a NUL"))?;
    let features = CString::new(features_for(arch)).expect("static feature string");

    unsafe {
        let mut target = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        if LLVMGetTargetFromTriple(TRIPLE.as_ptr(), &mut target, &mut error) != 0 {
            let message = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            return Err(message);
        }

        // `LLVMRelocPIC` is required: an AMD code object is a shared object, and the
        // default reloc model emits relocations LLD cannot resolve into an `ET_DYN`.
        let tm = LLVMCreateTargetMachine(
            target,
            TRIPLE.as_ptr(),
            cpu.as_ptr(),
            features.as_ptr(),
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            LLVMRelocMode::LLVMRelocPIC,
            LLVMCodeModel::LLVMCodeModelDefault,
        );
        if tm.is_null() {
            return Err(format!("no target machine for '{arch}'"));
        }

        let (ctx, module) = match parse_ir(ir) {
            Ok(parsed) => parsed,
            Err(err) => {
                LLVMDisposeTargetMachine(tm);
                return Err(err);
            }
        };

        // Layout from the target machine, so the two can never drift apart.
        let layout = LLVMCreateTargetDataLayout(tm);
        LLVMSetModuleDataLayout(module, layout);
        LLVMDisposeTargetData(layout);

        let result = run_pipeline_and_emit(module, tm, want_asm);

        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        LLVMDisposeTargetMachine(tm);
        result
    }
}

/// Optimizes `module` for `tm` and emits the object (and optionally assembly).
/// Split out so the caller can dispose its handles on every path.
///
/// # Safety
/// `module` and `tm` must be live LLVM handles.
unsafe fn run_pipeline_and_emit(
    module: llvm_sys::prelude::LLVMModuleRef,
    tm: llvm_sys::target_machine::LLVMTargetMachineRef,
    want_asm: bool,
) -> Result<(Vec<u8>, Option<String>), String> {
    use llvm_sys::error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage};
    use llvm_sys::target_machine::LLVMCodeGenFileType;
    use llvm_sys::transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    };

    unsafe {
        let options = LLVMCreatePassBuilderOptions();
        let err = LLVMRunPasses(module, PASS_PIPELINE.as_ptr(), tm, options);
        LLVMDisposePassBuilderOptions(options);
        if !err.is_null() {
            let c_msg = LLVMGetErrorMessage(err);
            let msg = CStr::from_ptr(c_msg).to_string_lossy().into_owned();
            LLVMDisposeErrorMessage(c_msg);
            return Err(msg);
        }

        let asm = if want_asm {
            let bytes = emit(module, tm, LLVMCodeGenFileType::LLVMAssemblyFile)?;
            Some(String::from_utf8_lossy(&bytes).into_owned())
        } else {
            None
        };
        let object = emit(module, tm, LLVMCodeGenFileType::LLVMObjectFile)?;
        Ok((object, asm))
    }
}

/// One `LLVMTargetMachineEmitToMemoryBuffer` call, copied out and disposed.
///
/// # Safety
/// `module` and `tm` must be live LLVM handles.
unsafe fn emit(
    module: llvm_sys::prelude::LLVMModuleRef,
    tm: llvm_sys::target_machine::LLVMTargetMachineRef,
    kind: llvm_sys::target_machine::LLVMCodeGenFileType,
) -> Result<Vec<u8>, String> {
    use llvm_sys::core::{
        LLVMDisposeMemoryBuffer, LLVMDisposeMessage, LLVMGetBufferSize, LLVMGetBufferStart,
    };
    use llvm_sys::target_machine::LLVMTargetMachineEmitToMemoryBuffer;

    unsafe {
        let mut buffer = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        if LLVMTargetMachineEmitToMemoryBuffer(tm, module, kind, &mut error, &mut buffer) != 0 {
            let message = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            return Err(message);
        }
        let start = LLVMGetBufferStart(buffer) as *const u8;
        let len = LLVMGetBufferSize(buffer);
        let bytes = std::slice::from_raw_parts(start, len).to_vec();
        LLVMDisposeMemoryBuffer(buffer);
        Ok(bytes)
    }
}

/// Parses textual `ir` into a fresh context, returning both so the caller can
/// dispose them.
///
/// # Safety
/// The returned context and module are owned by the caller.
unsafe fn parse_ir(
    ir: &str,
) -> Result<
    (
        llvm_sys::prelude::LLVMContextRef,
        llvm_sys::prelude::LLVMModuleRef,
    ),
    String,
> {
    use llvm_sys::core::{
        LLVMContextCreate, LLVMContextDispose, LLVMCreateMemoryBufferWithMemoryRangeCopy,
        LLVMDisposeMessage,
    };
    use llvm_sys::ir_reader::LLVMParseIRInContext2;

    unsafe {
        let ctx = LLVMContextCreate();
        let buffer = LLVMCreateMemoryBufferWithMemoryRangeCopy(
            ir.as_ptr() as *const _,
            ir.len(),
            c"kernel".as_ptr(),
        );
        let mut module = std::ptr::null_mut();
        let mut parse_err = std::ptr::null_mut();
        // `LLVMParseIRInContext2` consumes the buffer, on failure included.
        if LLVMParseIRInContext2(ctx, buffer, &mut module, &mut parse_err) != 0 {
            let msg = CStr::from_ptr(parse_err).to_string_lossy().into_owned();
            LLVMDisposeMessage(parse_err);
            LLVMContextDispose(ctx);
            return Err(msg);
        }
        Ok((ctx, module))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The finalized module carries everything the AMDGPU backend needs.
    #[test]
    fn finalize_sets_triple_callconv_and_arch() {
        let ir = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}
"#;
        let finalized = finalize_ir(ir, "k", "gfx1201", 64).unwrap();
        assert!(
            finalized.contains(r#"target triple = "amdgcn-amd-amdhsa""#),
            "{finalized}"
        );
        assert!(finalized.contains("amdgpu_kernel"), "{finalized}");
        assert!(
            finalized.contains(r#""target-cpu"="gfx1201""#),
            "{finalized}"
        );
        assert!(
            finalized.contains("amdhsa_code_object_version"),
            "{finalized}"
        );
    }

    /// Codegen produces a relocatable ELF, and LLD turns it into the `ET_DYN`
    /// shared object `hipModuleLoadData` requires. `e_type` is the 16-bit LE
    /// field at offset 16: 1 = `ET_REL`, 3 = `ET_DYN`.
    #[test]
    fn emits_a_linked_shared_object() {
        let ir = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}
"#;
        let finalized = finalize_ir(ir, "k", "gfx1201", 64).unwrap();
        let (object, asm) = compile_to_object(&finalized, "gfx1201", true).unwrap();
        assert_eq!(&object[..4], b"\x7fELF");
        assert_eq!(
            u16::from_le_bytes([object[16], object[17]]),
            1,
            "codegen gives ET_REL"
        );
        assert_eq!(object[7], 64, "EI_OSABI is ELFOSABI_AMDGPU_HSA");
        assert_eq!(object[8], 3, "EI_ABIVERSION is code object v5");
        assert_eq!(
            u16::from_le_bytes([object[18], object[19]]),
            0xe0,
            "EM_AMDGPU"
        );
        assert_eq!(
            u32::from_le_bytes(object[48..52].try_into().unwrap()) & 0xff,
            0x4e,
            "e_flags names gfx1201"
        );
        assert!(
            asm.unwrap().contains("amdhsa.kernels"),
            "assembly should carry HSA metadata"
        );

        let code = crate::amdgpu::lld::link_relocatable(&object, "k").unwrap();
        assert_eq!(&code[..4], b"\x7fELF");
        assert_eq!(
            u16::from_le_bytes([code[16], code[17]]),
            3,
            "lld must give ET_DYN"
        );
    }
}
