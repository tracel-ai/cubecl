//! Compiling the LLVM dialect to an AMD code object.

use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron_llvm::attributes::set_data_layout;
use pliron_llvm::llvm_sys::core::LLVMContext;
use pliron_llvm::to_llvm_ir;
use std::ffi::{CStr, CString};
use std::sync::Once;

use crate::amdgpu::device_libs::{DeviceLibs, link_device_libs};
use crate::amdgpu::lld::link_relocatable;
use crate::amdgpu::ocml::redirect_intrinsics_to_ocml;
use crate::amdgpu::printf::lower_printf_to_hostcall;
use crate::shared::AmdGpuModule;
use cubecl_core::ir::amd::GfxArch;
use cubecl_core::ir::attributes::BufferIOAttr;
use cubecl_environment::bytes::Bytes;

/// The HSA target triple; the specific device is the `-mcpu`, not the triple.
const TRIPLE: &CStr = c"amdgcn-amd-amdhsa";

/// AMDGPU's private stack address space used by `alloca` instructions.
const DATA_LAYOUT: &str = "A5";

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
fn features_for(arch: &GfxArch) -> &'static str {
    if arch.plane_dim() == Some(32) {
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
    arch: &GfxArch,
    cube_dim: u32,
    shared_memory_size: usize,
    io: Vec<BufferIOAttr>,
) -> Result<AmdGpuModule, String> {
    let llvm_ctx = LLVMContext::default();

    set_data_layout(ctx, module, DATA_LAYOUT.to_string());
    let llvm_module =
        to_llvm_ir::convert_module(ctx, &llvm_ctx, module).map_err(|err| err.to_string())?;

    let ir = finalize_ir(&llvm_module.to_string(), entrypoint, arch, cube_dim)?;
    let want_asm = std::env::var_os("CUBECL_DEBUG_PLIRON").is_some();

    let (object, asm) = compile_to_object(&ir, arch, want_asm)?;

    #[cfg(feature = "pliron-dump")]
    if let Some(dir) = crate::cpu::jit::engine::ir_dump_path(entrypoint) {
        let _ = std::fs::write(dir.join("amdgpu.ll"), &ir);
        if let Some(asm) = &asm {
            let _ = std::fs::write(dir.join("amdgpu.s"), asm);
        }
    }

    let code_object = Bytes::from_bytes_vec(link_relocatable(&object, entrypoint)?);

    Ok(AmdGpuModule {
        code_object,
        entrypoint: entrypoint.to_string(),
        ir,
        asm,
        shared_memory_size,
        io,
    })
}

/// Stamps `ir` with what the AMDGPU backend keys off: the HSA triple, the
/// `amdgpu_kernel` calling convention on `entrypoint`, the subtarget attributes,
/// and the code object version.
fn finalize_ir(
    ir: &str,
    entrypoint: &str,
    arch: &GfxArch,
    cube_dim: u32,
) -> Result<String, String> {
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
        ("target-cpu", arch.name()),
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
    arch: &GfxArch,
    want_asm: bool,
) -> Result<(Vec<u8>, Option<String>), String> {
    use llvm_sys::core::{LLVMContextDispose, LLVMDisposeMessage, LLVMDisposeModule};
    use llvm_sys::target::{LLVMDisposeTargetData, LLVMSetModuleDataLayout};
    use llvm_sys::target_machine::{
        LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetDataLayout, LLVMCreateTargetMachine,
        LLVMDisposeTargetMachine, LLVMGetTargetFromTriple, LLVMRelocMode,
    };

    init_amdgpu();

    let cpu =
        CString::new(arch.name()).map_err(|_| format!("arch '{}' contains a NUL", arch.name()))?;
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
            return Err(format!("no target machine for '{}'", arch.name()));
        }

        let (ctx, module) = match parse_ir(ir) {
            Ok(parsed) => parsed,
            Err(err) => {
                LLVMDisposeTargetMachine(tm);
                return Err(err);
            }
        };

        // Layout from the target machine, so the two can never drift apart. It has to be
        // set before the device libraries are linked, since they carry their own.
        let layout = LLVMCreateTargetDataLayout(tm);
        LLVMSetModuleDataLayout(module, layout);
        LLVMDisposeTargetData(layout);

        let result = lower_to_device_libs(module, arch)
            .and_then(|()| run_pipeline_and_emit(module, tm, want_asm));

        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        LLVMDisposeTargetMachine(tm);
        result
    }
}

/// Rewrites what the AMDGPU backend cannot handle on its own into calls to the `ROCm` device
/// libraries, and links those in.
///
/// # Safety
/// `module` must be a live LLVM module.
unsafe fn lower_to_device_libs(
    module: llvm_sys::prelude::LLVMModuleRef,
    arch: &GfxArch,
) -> Result<(), String> {
    unsafe {
        let needs = DeviceLibs {
            math: redirect_intrinsics_to_ocml(module)?,
            printf: lower_printf_to_hostcall(module),
        };

        if needs.any() {
            link_device_libs(module, arch, needs, CODE_OBJECT_VERSION)?;
        }
        Ok(())
    }
}

/// Runs the pass `pipeline` over `module`.
///
/// # Safety
/// `module` and `tm` must be live LLVM handles.
unsafe fn run_passes(
    module: llvm_sys::prelude::LLVMModuleRef,
    tm: llvm_sys::target_machine::LLVMTargetMachineRef,
    pipeline: &CStr,
) -> Result<(), String> {
    use llvm_sys::error::{LLVMDisposeErrorMessage, LLVMGetErrorMessage};
    use llvm_sys::transforms::pass_builder::{
        LLVMCreatePassBuilderOptions, LLVMDisposePassBuilderOptions, LLVMRunPasses,
    };

    unsafe {
        let options = LLVMCreatePassBuilderOptions();
        let err = LLVMRunPasses(module, pipeline.as_ptr(), tm, options);
        LLVMDisposePassBuilderOptions(options);
        if !err.is_null() {
            let c_msg = LLVMGetErrorMessage(err);
            let msg = CStr::from_ptr(c_msg).to_string_lossy().into_owned();
            LLVMDisposeErrorMessage(c_msg);
            return Err(msg);
        }
        Ok(())
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
    use llvm_sys::core::{LLVMCloneModule, LLVMDisposeModule};
    use llvm_sys::target_machine::LLVMCodeGenFileType;

    unsafe {
        run_passes(module, tm, PASS_PIPELINE)?;

        // Emission lowers the module in place, so a second file has to come off a copy.
        let asm = if want_asm {
            let copy = LLVMCloneModule(module);
            let bytes = emit(copy, tm, LLVMCodeGenFileType::LLVMAssemblyFile);
            LLVMDisposeModule(copy);
            Some(String::from_utf8_lossy(&bytes?).into_owned())
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

    /// The wave32 subtarget feature goes on the RDNA parts and nothing else. It is passed
    /// both as a function attribute and to the target machine, so a wrong answer here has
    /// every cross-lane lowering generating for the wrong wavefront width.
    #[test]
    fn only_the_rdna_parts_ask_for_wave32() {
        for name in ["gfx1201", "gfx1100", "gfx1030"] {
            assert_eq!(features_for(&GfxArch::parse(name)), WAVE32, "{name}");
        }
        for name in ["gfx90a", "gfx942", "gfx908"] {
            assert_eq!(features_for(&GfxArch::parse(name)), "", "{name}");
        }
    }

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
        let finalized = finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), 64).unwrap();
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

    /// The shared memory block reaches the code object as LDS: the slices become `ds_`
    /// accesses with their offset folded in, the generic pointers the rest of the pipeline
    /// works with are inferred away, and the barrier is the hardware's own.
    #[test]
    fn shared_memory_becomes_lds() {
        let ir = r#"
@cube_lds = external addrspace(3) global [0 x i8], align 16
declare void @llvm.amdgcn.s.barrier()
define void @k(ptr addrspace(1) %out, i32 %tid) {
entry:
  %slice = getelementptr i8, ptr addrspace(3) @cube_lds, i32 64
  %flat = addrspacecast ptr addrspace(3) %slice to ptr
  %idx = getelementptr float, ptr %flat, i32 %tid
  store float 1.0, ptr %idx, align 4
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %v = load float, ptr %idx, align 4
  store float %v, ptr addrspace(1) %out
  ret void
}
"#;
        let finalized = finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), 64).unwrap();
        let (object, asm) =
            compile_to_object(&finalized, &GfxArch::parse("gfx1201"), true).unwrap();
        assert_eq!(&object[..4], b"\x7fELF");

        let asm = asm.unwrap();
        assert!(
            asm.contains("ds_store"),
            "the write should reach LDS:\n{asm}"
        );
        assert!(asm.contains("ds_load"), "the read should reach LDS:\n{asm}");
        assert!(
            !asm.contains("flat_store") && !asm.contains("flat_load"),
            "the generic pointers should be inferred away:\n{asm}"
        );
        assert!(asm.contains("s_barrier"), "the cube barrier:\n{asm}");

        // Dynamic, so the block costs the code object nothing and arrives as `sharedMemBytes`.
        assert!(
            asm.contains(".group_segment_fixed_size: 0"),
            "the block should be sized at launch, not baked in:\n{asm}"
        );

        crate::amdgpu::lld::link_relocatable(&object, "k").unwrap();
    }

    /// The matrix instruction reaches the code object, in the shape each generation asks for:
    /// RDNA4 splits `k` between the halves of the wave and takes half the A/B fragment RDNA3
    /// does. Getting the fragment width wrong fails to select rather than computing the wrong
    /// answer, so this pins both.
    #[test]
    fn wmma_reaches_the_code_object() {
        for (name, ab) in [("gfx1201", "<8 x half>"), ("gfx1100", "<16 x half>")] {
            let arch = GfxArch::parse(name);
            let width = if ab.starts_with("<8") {
                "v8f16"
            } else {
                "v16f16"
            };
            let ir = format!(
                r#"
declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.{width}({ab}, {ab}, <8 x float>)
define void @k(ptr addrspace(1) %out, {ab} %a, {ab} %b, <8 x float> %c) {{
entry:
  %d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.{width}({ab} %a, {ab} %b, <8 x float> %c)
  store <8 x float> %d, ptr addrspace(1) %out
  ret void
}}
"#
            );
            let finalized = finalize_ir(&ir, "k", &arch, 32).unwrap();
            let (object, asm) = compile_to_object(&finalized, &arch, true).unwrap();
            assert_eq!(&object[..4], b"\x7fELF");

            let asm = asm.unwrap();
            assert!(
                asm.contains("v_wmma_f32_16x16x16_f16"),
                "{name} should reach the matrix instruction:\n{asm}"
            );

            crate::amdgpu::lld::link_relocatable(&object, "k").unwrap();
        }
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
        let finalized = finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), 64).unwrap();
        let (object, asm) =
            compile_to_object(&finalized, &GfxArch::parse("gfx1201"), true).unwrap();
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
