//! PTX code generation.

use crate::{
    nvptx::{
        libdevice::{Libdevice, link_libdevice},
        printf::lower_printf_to_vprintf,
        ptx_version::PtxVersion,
    },
    prelude::{BufferIOAttr, Context, ModuleOp},
    shared::{NvptxModule, math_library::redirect_intrinsics},
};
use cubecl_core::ir::nvidia::SmArch;
use pliron_llvm::{llvm_sys::core::LLVMContext, to_llvm_ir};
use std::{
    ffi::{CStr, CString},
    sync::Once,
};

const TRIPLE: &CStr = c"nvptx64-nvidia-cuda";

/// LLVM calling convention for PTX kernel entry points.
const PTX_KERNEL_CC: u32 = 71;

const PASS_PIPELINE: &CStr = c"default<O3>";

static INIT_NVPTX: Once = Once::new();

fn init_nvptx() {
    INIT_NVPTX.call_once(|| unsafe {
        llvm_sys::target::LLVMInitializeNVPTXTargetInfo();
        llvm_sys::target::LLVMInitializeNVPTXTarget();
        llvm_sys::target::LLVMInitializeNVPTXTargetMC();
        llvm_sys::target::LLVMInitializeNVPTXAsmPrinter();
    });
}

/// Kernel metadata layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetadataParams {
    /// Scalars, static metadata, shapes and strides in one buffer.
    Buffer,
    /// Scalars and static metadata passed by value. Shapes and strides use a
    /// separate buffer when `dynamic_buffer` is set.
    GridConstant { bytes: usize, dynamic_buffer: bool },
}

impl MetadataParams {
    fn count(self) -> u32 {
        match self {
            MetadataParams::Buffer => 1,
            MetadataParams::GridConstant { dynamic_buffer, .. } => 1 + dynamic_buffer as u32,
        }
    }
}

/// Kernel entry point requirements.
pub struct NvptxEntry {
    /// Maximum units per cube.
    pub cube_dim: u32,
    /// Shared memory required per launch, in bytes.
    pub shared_memory_size: usize,
    /// Buffer access modes in binding order.
    pub io: Vec<BufferIOAttr>,
    /// Metadata parameter layout.
    pub metadata: MetadataParams,
}

pub fn emit_ptx(
    ctx: &Context,
    module: ModuleOp,
    entrypoint: &str,
    arch: &SmArch,
    ptx_version: Option<PtxVersion>,
    entry: NvptxEntry,
) -> Result<NvptxModule, String> {
    let llvm_ctx = LLVMContext::default();

    let llvm_module =
        to_llvm_ir::convert_module(ctx, &llvm_ctx, module).map_err(|err| err.to_string())?;

    let ir = finalize_ir(&llvm_module.to_string(), entrypoint, arch, &entry)?;
    let ptx = compile_to_ptx(&ir, arch, ptx_version)?;

    #[cfg(feature = "pliron-dump")]
    if let Some(dir) = crate::cpu::jit::engine::ir_dump_path(entrypoint) {
        let _ = std::fs::write(dir.join("nvptx.ll"), &ir);
        let _ = std::fs::write(dir.join("nvptx.ptx"), &ptx);
    }

    Ok(NvptxModule {
        ptx: as_c_chars(&ptx),
        entrypoint: entrypoint.to_string(),
        ir,
        shared_memory_size: entry.shared_memory_size,
        io: entry.io,
    })
}

fn finalize_ir(
    ir: &str,
    entrypoint: &str,
    arch: &SmArch,
    entry: &NvptxEntry,
) -> Result<String, String> {
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMContextDispose, LLVMCreateStringAttribute, LLVMDisposeMessage,
        LLVMDisposeModule, LLVMGetNamedFunction, LLVMPrintModuleToString, LLVMSetFunctionCallConv,
        LLVMSetTarget,
    };

    let name = CString::new(entrypoint)
        .map_err(|_| format!("kernel name '{entrypoint}' contains a NUL"))?;

    let target_cpu = arch.target_cpu();
    // Launch bounds limit register use for the requested cube size.
    let max_threads = entry.cube_dim.to_string();
    let attributes = [
        ("target-cpu", target_cpu.as_str()),
        ("nvvm.maxntid", max_threads.as_str()),
    ];

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
        LLVMSetFunctionCallConv(func, PTX_KERNEL_CC);

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

        annotate_buffer_params(ctx, func, &entry.io, entry.metadata);
        if let MetadataParams::GridConstant { bytes, .. } = entry.metadata {
            mark_info_param_byval(ctx, func, bytes);
        }

        let c_ir = LLVMPrintModuleToString(module);
        let finalized = CStr::from_ptr(c_ir).to_string_lossy().into_owned();
        LLVMDisposeMessage(c_ir);
        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        Ok(finalized)
    }
}

/// Alignment required by the host metadata layout.
const INFO_PARAM_ALIGN: u32 = 8;

/// # Safety
/// `func` must be a live function in `ctx` whose last parameter is the info pointer the entry
/// ABI lowering appended.
unsafe fn mark_info_param_byval(
    ctx: llvm_sys::prelude::LLVMContextRef,
    func: llvm_sys::prelude::LLVMValueRef,
    bytes: usize,
) {
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMArrayType2, LLVMCountParams, LLVMCreateEnumAttribute,
        LLVMCreateTypeAttribute, LLVMGetEnumAttributeKindForName, LLVMInt8TypeInContext,
    };

    unsafe {
        let enum_kind =
            |name: &str| LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
        let (byval, align) = (enum_kind("byval"), enum_kind("align"));
        assert!(
            byval != 0 && align != 0,
            "this LLVM has no `byval` or `align` attribute, so the grid-constant parameter \
             cannot be declared"
        );

        let index = LLVMCountParams(func);
        let block = LLVMArrayType2(LLVMInt8TypeInContext(ctx), bytes as u64);
        LLVMAddAttributeAtIndex(func, index, LLVMCreateTypeAttribute(ctx, byval, block));
        LLVMAddAttributeAtIndex(
            func,
            index,
            LLVMCreateEnumAttribute(ctx, align, INFO_PARAM_ALIGN as u64),
        );
    }
}

/// # Safety
/// `func` must be a live function in `ctx` whose parameters are the buffers in binding order
/// followed by the metadata pointer.
unsafe fn annotate_buffer_params(
    ctx: llvm_sys::prelude::LLVMContextRef,
    func: llvm_sys::prelude::LLVMValueRef,
    io: &[BufferIOAttr],
    metadata: MetadataParams,
) {
    use llvm_sys::LLVMTypeKind;
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMCountParams, LLVMCreateEnumAttribute,
        LLVMGetEnumAttributeKindForName, LLVMGetParam, LLVMGetTypeKind, LLVMTypeOf,
    };

    unsafe {
        let enum_attr = |index: u32, name: &str| {
            let kind = LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
            if kind == 0 {
                return;
            }
            let attribute = LLVMCreateEnumAttribute(ctx, kind, 0);
            LLVMAddAttributeAtIndex(func, index, attribute);
        };

        // Atomic loads must retain coherent memory access.
        let may_say_readonly = !reads_atomically(func);

        let params = LLVMCountParams(func);
        let first_metadata = params.saturating_sub(metadata.count());
        for param in 0..params {
            if LLVMGetTypeKind(LLVMTypeOf(LLVMGetParam(func, param)))
                != LLVMTypeKind::LLVMPointerTypeKind
            {
                continue;
            }

            let index = param + 1;
            enum_attr(index, "noalias");

            let read_only = param >= first_metadata
                || io
                    .get(param as usize)
                    .is_some_and(|attr| *attr == BufferIOAttr::ReadOnly);
            if read_only && may_say_readonly {
                enum_attr(index, "readonly");
            }
        }
    }
}

/// # Safety
/// `func` must be a live LLVM function.
unsafe fn reads_atomically(func: llvm_sys::prelude::LLVMValueRef) -> bool {
    use llvm_sys::LLVMAtomicOrdering;
    use llvm_sys::core::{
        LLVMGetFirstBasicBlock, LLVMGetFirstInstruction, LLVMGetNextBasicBlock,
        LLVMGetNextInstruction, LLVMGetOrdering, LLVMIsALoadInst,
    };

    unsafe {
        let mut block = LLVMGetFirstBasicBlock(func);
        while !block.is_null() {
            let mut inst = LLVMGetFirstInstruction(block);
            while !inst.is_null() {
                if !LLVMIsALoadInst(inst).is_null()
                    && LLVMGetOrdering(inst) != LLVMAtomicOrdering::LLVMAtomicOrderingNotAtomic
                {
                    return true;
                }
                inst = LLVMGetNextInstruction(inst);
            }
            block = LLVMGetNextBasicBlock(block);
        }
        false
    }
}

/// `ptx_version` is `None` for LLVM's default, the oldest the architecture accepts.
fn compile_to_ptx(
    ir: &str,
    arch: &SmArch,
    ptx_version: Option<PtxVersion>,
) -> Result<String, String> {
    use llvm_sys::core::{LLVMContextDispose, LLVMDisposeMessage, LLVMDisposeModule};
    use llvm_sys::target::{LLVMDisposeTargetData, LLVMSetModuleDataLayout};
    use llvm_sys::target_machine::{
        LLVMCodeGenOptLevel, LLVMCodeModel, LLVMCreateTargetDataLayout, LLVMCreateTargetMachine,
        LLVMDisposeTargetMachine, LLVMGetTargetFromTriple, LLVMRelocMode,
    };

    init_nvptx();

    let target_cpu = arch.target_cpu();
    let cpu = CString::new(target_cpu.clone())
        .map_err(|_| format!("arch '{target_cpu}' contains a NUL"))?;
    let features = ptx_version
        .map(PtxVersion::target_feature)
        .unwrap_or_default();

    unsafe {
        let mut target = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        if LLVMGetTargetFromTriple(TRIPLE.as_ptr(), &mut target, &mut error) != 0 {
            let message = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            return Err(message);
        }

        let tm = LLVMCreateTargetMachine(
            target,
            TRIPLE.as_ptr(),
            cpu.as_ptr(),
            features.as_ptr(),
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            LLVMRelocMode::LLVMRelocDefault,
            LLVMCodeModel::LLVMCodeModelDefault,
        );
        if tm.is_null() {
            return Err(format!("no target machine for '{target_cpu}'"));
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

        let result = lower_to_device_libs(module)
            .and_then(|()| run_passes(module, tm, PASS_PIPELINE))
            .and_then(|()| emit_assembly(module, tm))
            .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());

        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        LLVMDisposeTargetMachine(tm);
        result
    }
}

/// # Safety
/// `module` must be a live LLVM module.
unsafe fn lower_to_device_libs(module: llvm_sys::prelude::LLVMModuleRef) -> Result<(), String> {
    unsafe {
        lower_printf_to_vprintf(module)?;
        if redirect_intrinsics(module, &Libdevice)? {
            link_libdevice(module)?;
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
unsafe fn emit_assembly(
    module: llvm_sys::prelude::LLVMModuleRef,
    tm: llvm_sys::target_machine::LLVMTargetMachineRef,
) -> Result<Vec<u8>, String> {
    use llvm_sys::core::{
        LLVMDisposeMemoryBuffer, LLVMDisposeMessage, LLVMGetBufferSize, LLVMGetBufferStart,
    };
    use llvm_sys::target_machine::{LLVMCodeGenFileType, LLVMTargetMachineEmitToMemoryBuffer};

    unsafe {
        let mut buffer = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        if LLVMTargetMachineEmitToMemoryBuffer(
            tm,
            module,
            LLVMCodeGenFileType::LLVMAssemblyFile,
            &mut error,
            &mut buffer,
        ) != 0
        {
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

/// NUL-terminated PTX for the CUDA driver.
fn as_c_chars(ptx: &str) -> Vec<std::ffi::c_char> {
    let mut bytes: Vec<std::ffi::c_char> =
        ptx.bytes().map(|byte| byte as std::ffi::c_char).collect();
    bytes.push(0);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turing_reaches_its_tensor_core_instructions() {
        let ir = r#"
declare i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16(ptr addrspace(3))
declare { float, float, float, float } @llvm.nvvm.mma.m16n8k8.row.col.f32.f32(<2 x half>, <2 x half>, <2 x half>, float, float, float, float)
define void @k(ptr addrspace(1) %out, ptr addrspace(3) %tile, <2 x half> %a) {
entry:
  %b = call i32 @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x1.b16(ptr addrspace(3) %tile)
  %bh = bitcast i32 %b to <2 x half>
  %d = call { float, float, float, float } @llvm.nvvm.mma.m16n8k8.row.col.f32.f32(<2 x half> %a, <2 x half> %a, <2 x half> %bh, float 0.0, float 0.0, float 0.0, float 0.0)
  %d0 = extractvalue { float, float, float, float } %d, 0
  store float %d0, ptr addrspace(1) %out
  ret void
}
"#;
        // The oldest driver with these instructions, and a current one. LLVM's own default
        // (6.3) aborts in instruction selection here rather than returning an error.
        for driver in [10020, 12080] {
            let ptx_version = PtxVersion::for_driver(driver);
            let ptx = compile_to_ptx(ir, &SmArch::new(75, true), ptx_version).unwrap();
            assert!(
                ptx.contains("ldmatrix.sync.aligned"),
                "CUDA {driver}:\n{ptx}"
            );
            assert!(
                ptx.contains("mma.sync.aligned.m16n8k8"),
                "CUDA {driver}:\n{ptx}"
            );
        }
    }

    #[test]
    fn the_newest_driver_version_is_one_this_llvm_emits() {
        let ir = r#"
define void @k() {
entry:
  ret void
}
"#;
        let ptx_version = PtxVersion::for_driver(i32::MAX);
        let ptx = compile_to_ptx(ir, &SmArch::new(75, true), ptx_version).unwrap();
        assert!(
            ptx.contains(".version 9.3"),
            "an unknown version is ignored, not refused:\n{ptx}"
        );
    }
}
