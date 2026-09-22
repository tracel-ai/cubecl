//! PTX code generation.

use crate::{
    nvptx::{
        libdevice::{Libdevice, link_libdevice},
        printf::lower_printf_to_vprintf,
        ptx_version::PtxVersion,
    },
    prelude::{BufferIOAttr, Context, ModuleOp},
    shared::{
        NvptxModule,
        buffer_params::annotate_buffer_params,
        llvm_module::{LlvmModule, TargetMachine},
        llvm_options::set_llvm_option,
        math_library::redirect_intrinsics,
    },
};
use cubecl_core::ir::{nvidia::SmArch, settings::Dim3};
use llvm_sys::target_machine::{LLVMCodeGenFileType, LLVMRelocMode};
use pliron_llvm::{llvm_sys::core::LLVMContext, to_llvm_ir};
use std::{ffi::CStr, sync::Once};

const TRIPLE: &CStr = c"nvptx64-nvidia-cuda";

/// LLVM calling convention for PTX kernel entry points.
const PTX_KERNEL_CC: u32 = 71;

const PASS_PIPELINE: &CStr = c"default<O3>";

static INIT_NVPTX: Once = Once::new();

/// `atom.add.f32` on global memory flushes denormals, so LLVM keeps it only for a function
/// that flushes them too and expands every other float `atomicrmw fadd` to a CAS loop. The
/// kernels here keep denormals in ordinary arithmetic, as NVRTC does by default, while NVRTC's
/// `atomicAdd(float*)` is the native instruction all the same. This option makes the same
/// trade: a denormal lost in an atomic sum, for an atomic that is not a retry loop.
const ALLOW_FTZ_ATOMICS: &CStr = c"nvptx-allow-ftz-atomics";

fn init_nvptx() {
    INIT_NVPTX.call_once(|| unsafe {
        llvm_sys::target::LLVMInitializeNVPTXTargetInfo();
        llvm_sys::target::LLVMInitializeNVPTXTarget();
        llvm_sys::target::LLVMInitializeNVPTXTargetMC();
        llvm_sys::target::LLVMInitializeNVPTXAsmPrinter();

        // An LLVM without the option still compiles correct kernels, with CAS-loop atomics;
        // `a_float_atomic_add_is_the_native_instruction` is what notices.
        set_llvm_option(ALLOW_FTZ_ATOMICS, c"true");
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
    /// Units per cube along each axis.
    pub cube_dim: Dim3,
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
    let converted =
        to_llvm_ir::convert_module(ctx, &llvm_ctx, module).map_err(|err| err.to_string())?;

    let module = LlvmModule::parse(&converted.to_string())?;
    finalize(&module, entrypoint, arch, &entry)?;
    let ir = module.print();
    let ptx = compile(module, arch, ptx_version)?;

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

/// Stamps the target and the entry point's calling convention and attributes on `module`.
fn finalize(
    module: &LlvmModule,
    entrypoint: &str,
    arch: &SmArch,
    entry: &NvptxEntry,
) -> Result<(), String> {
    use llvm_sys::core::LLVMSetFunctionCallConv;

    let target_cpu = arch.target_cpu();
    // The cube dimensions are fixed when a kernel compiles, so the launch bounds are exact:
    // they limit register use to what the cube needs, and they bound each `tid` register, so an
    // axis of one unit reads as zero rather than as a register.
    let Dim3 { x, y, z } = entry.cube_dim;
    let threads = format!("{x},{y},{z}");
    let attributes = [
        ("target-cpu", target_cpu.as_str()),
        ("nvvm.reqntid", threads.as_str()),
    ];

    module.set_triple(TRIPLE);
    let func = module.entry_point(entrypoint)?;
    // SAFETY: `func` is a function of `module`, whose parameters the entry ABI lowering laid
    // out as the buffers in binding order followed by the metadata.
    unsafe {
        LLVMSetFunctionCallConv(func, PTX_KERNEL_CC);
        module.add_function_attributes(func, &attributes);
        annotate_buffer_params(module.context(), func, &entry.io, entry.metadata.count());
        if let MetadataParams::GridConstant { bytes, .. } = entry.metadata {
            mark_info_param_byval(module.context(), func, bytes);
        }
    }
    Ok(())
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

/// `ptx_version` is `None` for LLVM's default, the oldest the architecture accepts.
fn compile(
    module: LlvmModule,
    arch: &SmArch,
    ptx_version: Option<PtxVersion>,
) -> Result<String, String> {
    init_nvptx();

    let features = ptx_version
        .map(PtxVersion::target_feature)
        .unwrap_or_default();
    let machine = TargetMachine::new(
        TRIPLE,
        &arch.target_cpu(),
        &features,
        LLVMRelocMode::LLVMRelocDefault,
    )?;
    machine.set_data_layout(&module);

    // SAFETY: the module is live for the call.
    unsafe { lower_to_device_libs(module.raw())? };
    module.run_passes(PASS_PIPELINE, Some(&machine))?;
    let bytes = machine.emit(module, LLVMCodeGenFileType::LLVMAssemblyFile)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
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

    fn compile_to_ptx(
        ir: &str,
        arch: &SmArch,
        ptx_version: Option<PtxVersion>,
    ) -> Result<String, String> {
        compile(LlvmModule::parse(ir)?, arch, ptx_version)
    }

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
    fn a_float_atomic_add_is_the_native_instruction() {
        let ir = r#"
define void @k(ptr addrspace(1) %p, float %v, ptr addrspace(1) %o) {
entry:
  %r = atomicrmw fadd ptr addrspace(1) %p, float %v syncscope("device") monotonic, align 4
  store float %r, ptr addrspace(1) %o
  ret void
}
"#;
        let ptx =
            compile_to_ptx(ir, &SmArch::new(60, false), PtxVersion::for_driver(12080)).unwrap();
        assert!(ptx.contains("atom.gpu.global.add.f32"), "{ptx}");
        assert!(!ptx.contains(".cas."), "no CAS loop:\n{ptx}");
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
