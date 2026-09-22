//! AMDGPU code generation.

use crate::{
    amdgpu::{
        device_libs::{DeviceLibs, link_device_libs},
        lld::link_relocatable,
        ocml::Ocml,
        printf::lower_printf_to_hostcall,
    },
    prelude::{BufferIOAttr, Context, ModuleOp},
    shared::{
        AmdGpuModule, buffer_params::annotate_buffer_params, math_library::redirect_intrinsics,
    },
};
use cubecl_core::ir::amd::GfxArch;
use cubecl_environment::bytes::Bytes;
use pliron_llvm::{attributes::set_data_layout, llvm_sys::core::LLVMContext, to_llvm_ir};
use std::{
    ffi::{CStr, CString},
    sync::Once,
};

const TRIPLE: &CStr = c"amdgcn-amd-amdhsa";

const DATA_LAYOUT: &str = "A5";

/// HSA code object version.
const CODE_OBJECT_VERSION: u32 = 500;

const AMDGPU_KERNEL_CC: u32 = 91;

/// Wave32 feature for RDNA devices.
const WAVE32: &str = "+wavefrontsize32";

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

fn features_for(arch: &GfxArch) -> &'static str {
    if arch.plane_dim() == Some(32) {
        WAVE32
    } else {
        ""
    }
}

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

    let ir = finalize_ir(&llvm_module.to_string(), entrypoint, arch, cube_dim, &io)?;
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

/// The metadata pointer `KernargArgs` appends after the buffers.
const METADATA_PARAMS: u32 = 1;

fn finalize_ir(
    ir: &str,
    entrypoint: &str,
    arch: &GfxArch,
    cube_dim: u32,
    io: &[BufferIOAttr],
) -> Result<String, String> {
    use llvm_sys::LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorError;
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMAddModuleFlag, LLVMConstInt, LLVMContextDispose,
        LLVMCreateStringAttribute, LLVMDisposeMessage, LLVMDisposeModule, LLVMGetNamedFunction,
        LLVMInt32TypeInContext, LLVMPrintModuleToString, LLVMSetFunctionCallConv, LLVMSetTarget,
        LLVMValueAsMetadata,
    };

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

        annotate_buffer_params(ctx, func, io, METADATA_PARAMS);
        mark_atomics_device_local(ctx, func);

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

/// What every atomic here may assume about the memory it touches, as the metadata the AMDGPU
/// backend reads.
///
/// Kernel buffers are device allocations: coarse-grained, and never another device's memory,
/// so no atomic has to stay correct against a concurrent host or peer access. Without saying
/// so, the backend expands float atomics to CAS loops on the parts whose native instruction
/// is not coherent for fine-grained memory (RDNA3, CDNA2). An f32 add also ignores the
/// denormal mode, where the native instruction flushes, which is the trade NVRTC's
/// `atomicAdd(float*)` makes and the NVPTX target makes too.
const DEVICE_LOCAL_ATOMIC: [&str; 2] = ["amdgpu.no.fine.grained.memory", "amdgpu.no.remote.memory"];
const DENORMAL_AGNOSTIC_ATOMIC: &str = "amdgpu.ignore.denormal.mode";

/// # Safety
/// `func` must be a live function in `ctx`.
unsafe fn mark_atomics_device_local(
    ctx: llvm_sys::prelude::LLVMContextRef,
    func: llvm_sys::prelude::LLVMValueRef,
) {
    use llvm_sys::LLVMAtomicRMWBinOp;
    use llvm_sys::core::{
        LLVMGetAtomicRMWBinOp, LLVMGetFirstBasicBlock, LLVMGetFirstInstruction,
        LLVMGetMDKindIDInContext, LLVMGetNextBasicBlock, LLVMGetNextInstruction,
        LLVMIsAAtomicRMWInst, LLVMMDNodeInContext2, LLVMMetadataAsValue, LLVMSetMetadata,
    };

    unsafe {
        let mark = |inst, name: &str| {
            let kind = LLVMGetMDKindIDInContext(ctx, name.as_ptr() as *const _, name.len() as u32);
            let empty =
                LLVMMetadataAsValue(ctx, LLVMMDNodeInContext2(ctx, std::ptr::null_mut(), 0));
            LLVMSetMetadata(inst, kind, empty);
        };

        let mut block = LLVMGetFirstBasicBlock(func);
        while !block.is_null() {
            let mut inst = LLVMGetFirstInstruction(block);
            while !inst.is_null() {
                if !LLVMIsAAtomicRMWInst(inst).is_null() {
                    for name in DEVICE_LOCAL_ATOMIC {
                        mark(inst, name);
                    }
                    if LLVMGetAtomicRMWBinOp(inst) == LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpFAdd {
                        mark(inst, DENORMAL_AGNOSTIC_ATOMIC);
                    }
                }
                inst = LLVMGetNextInstruction(inst);
            }
            block = LLVMGetNextBasicBlock(block);
        }
    }
}

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

        // AMD code objects require position-independent code.
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

/// # Safety
/// `module` must be a live LLVM module.
unsafe fn lower_to_device_libs(
    module: llvm_sys::prelude::LLVMModuleRef,
    arch: &GfxArch,
) -> Result<(), String> {
    unsafe {
        let needs = DeviceLibs {
            math: redirect_intrinsics(module, &Ocml)?,
            printf: lower_printf_to_hostcall(module),
        };

        if needs.any() {
            link_device_libs(module, arch, needs, CODE_OBJECT_VERSION)?;
        }
        Ok(())
    }
}

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

        // Emission modifies the module, so each output needs its own copy.
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
        // `LLVMParseIRInContext2` takes ownership of the buffer, including on failure.
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

    #[test]
    fn only_the_rdna_parts_ask_for_wave32() {
        for name in ["gfx1201", "gfx1100", "gfx1030"] {
            assert_eq!(features_for(&GfxArch::parse(name)), WAVE32, "{name}");
        }
        for name in ["gfx90a", "gfx942", "gfx908"] {
            assert_eq!(features_for(&GfxArch::parse(name)), "", "{name}");
        }
    }

    #[test]
    fn finalize_sets_triple_callconv_and_arch() {
        let ir = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}
"#;
        let finalized = finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), 64, &[]).unwrap();
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

    #[test]
    fn shared_memory_becomes_lds() {
        let ir = r#"
@cube_shared = external addrspace(3) global [0 x i8], align 16
declare void @llvm.amdgcn.s.barrier()
define void @k(ptr addrspace(1) %out, i32 %tid) {
entry:
  %slice = getelementptr i8, ptr addrspace(3) @cube_shared, i32 64
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
        let finalized = finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), 64, &[]).unwrap();
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

        assert!(
            asm.contains(".group_segment_fixed_size: 0"),
            "the block should be sized at launch, not baked in:\n{asm}"
        );

        crate::amdgpu::lld::link_relocatable(&object, "k").unwrap();
    }

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
            let finalized = finalize_ir(&ir, "k", &arch, 32, &[]).unwrap();
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

    #[test]
    fn a_uniform_metadata_read_stays_scalar_across_stores() {
        // The bound is read on every iteration, after a store through another buffer: only
        // knowing the two do not alias lets it be hoisted and loaded once, into an SGPR.
        let ir = r#"
define void @k(ptr addrspace(1) %out, ptr addrspace(1) %info) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %len = load i32, ptr addrspace(1) %info, align 4
  %slot = getelementptr inbounds nuw i32, ptr addrspace(1) %out, i32 %i
  store i32 %i, ptr addrspace(1) %slot, align 4
  %next = add nuw i32 %i, 1
  %more = icmp ult i32 %next, %len
  br i1 %more, label %loop, label %exit
exit:
  ret void
}
"#;
        let arch = GfxArch::parse("gfx1201");
        let finalized = finalize_ir(ir, "k", &arch, 64, &[BufferIOAttr::WriteOnly]).unwrap();
        assert!(finalized.contains("noalias"), "{finalized}");
        assert!(
            finalized.contains("readonly"),
            "the metadata is read-only:\n{finalized}"
        );

        let (_, asm) = compile_to_object(&finalized, &arch, true).unwrap();
        let asm = asm.unwrap();
        assert!(
            asm.contains("s_load_b32"),
            "the bound is a scalar load:\n{asm}"
        );
        assert!(
            !asm.contains("global_load"),
            "the bound is not reloaded per iteration:\n{asm}"
        );
    }

    #[test]
    fn a_float_atomic_add_is_the_native_instruction() {
        let ir = r#"
define void @k(ptr addrspace(1) %p, float %v, ptr addrspace(1) %o) {
entry:
  %r = atomicrmw fadd ptr addrspace(1) %p, float %v syncscope("agent") monotonic, align 4
  store float %r, ptr addrspace(1) %o
  ret void
}
"#;
        // RDNA3 and CDNA2 are the parts that fall back to a CAS loop without the metadata.
        for name in ["gfx1100", "gfx90a", "gfx1201", "gfx942"] {
            let arch = GfxArch::parse(name);
            let finalized = finalize_ir(ir, "k", &arch, 64, &[]).unwrap();
            let (_, asm) = compile_to_object(&finalized, &arch, true).unwrap();
            let asm = asm.unwrap();
            assert!(asm.contains("global_atomic_add_f32"), "{name}:\n{asm}");
            assert!(!asm.contains("cmpswap"), "{name} has no CAS loop:\n{asm}");
        }
    }

    #[test]
    fn emits_a_linked_shared_object() {
        let ir = r#"
define void @k(ptr addrspace(1) %out) {
entry:
  store i32 7, ptr addrspace(1) %out, align 4
  ret void
}
"#;
        let finalized = finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), 64, &[]).unwrap();
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
