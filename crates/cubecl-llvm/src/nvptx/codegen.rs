//! PTX code generation.

use crate::{
    nvptx::{
        libdevice::{Libdevice, link_libdevice},
        printf::lower_printf_to_vprintf,
        ptx_version::PtxVersion,
        tensor_map::TensorMapParams,
    },
    prelude::{BufferIOAttr, Context, ModuleOp},
    shared::{
        NvptxModule,
        buffer_params::{AtomicReads, annotate_buffer_params},
        debug_info::{check_debug_info, convert_module},
        llvm_module::{EntryFunction, LlvmModule, TargetMachine, TargetSpec},
        llvm_options::{read_llvm_options, set_llvm_option},
        math_library::redirect_intrinsics,
    },
};
use cubecl_core::{
    codegen::KernelDump,
    ir::{
        nvidia::SmArch,
        settings::{DebugInfo, Dim3},
    },
};
use llvm_sys::target_machine::{LLVMCodeGenFileType, LLVMCodeModel, LLVMRelocMode};
use pliron_llvm::llvm_sys::core::LLVMContext;
use std::{ffi::CStr, fmt::Write, sync::Once};

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

/// The size a loop may grow to when LLVM unrolls it, raised from the `O3` default of 300 to what
/// keeps a tile kernel's inner loops unrolled, as NVVM leaves them. At 300 the packed staged
/// matmul keeps its fragment loops rolled, a quarter of NVRTC's `HMMA` per body, and runs a
/// quarter slower; at 1000 it is within a few percent of NVRTC, and the float staged matmul on
/// rows and tiles gains up to a tenth, with no family's fastest setting slower. LLVM holds it as
/// global state, so it is set once with the target, and applies to every kernel compiled after.
const UNROLL_THRESHOLD: (&CStr, &CStr) = (c"unroll-threshold", c"1000");

fn init_nvptx() {
    INIT_NVPTX.call_once(|| unsafe {
        llvm_sys::target::LLVMInitializeNVPTXTargetInfo();
        llvm_sys::target::LLVMInitializeNVPTXTarget();
        llvm_sys::target::LLVMInitializeNVPTXTargetMC();
        llvm_sys::target::LLVMInitializeNVPTXAsmPrinter();

        // An LLVM without the option still compiles correct kernels, with CAS-loop atomics;
        // `a_float_atomic_add_is_the_native_instruction` is what notices.
        let _ = set_llvm_option(ALLOW_FTZ_ATOMICS, c"true");
        let (name, value) = UNROLL_THRESHOLD;
        let _ = set_llvm_option(name, value);
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
    /// The global buffers the kernel's atomics read.
    pub atomic_reads: AtomicReads,
    /// Metadata parameter layout.
    pub metadata: MetadataParams,
    /// The parameters that are tensor maps.
    pub tensor_maps: TensorMapParams,
}

pub fn emit_ptx(
    ctx: &Context,
    module: ModuleOp,
    entrypoint: &str,
    arch: &SmArch,
    ptx_version: Option<PtxVersion>,
    entry: NvptxEntry,
    debug_info: DebugInfo,
) -> Result<NvptxModule, String> {
    let eff_arch = effective_arch(arch, ptx_version);
    let llvm_ctx = LLVMContext::default();
    let converted =
        convert_module(ctx, &llvm_ctx, module, debug_info, false).map_err(|err| err.to_string())?;

    let ir = name_inlined_functions(directives_only(converted.to_string(), debug_info));
    let module = LlvmModule::new(&ir)?;
    check_debug_info(&module, entrypoint, debug_info);
    finalize(&module, entrypoint, &eff_arch, &entry)?;
    let ir = module.print();
    // Written before the compile, so a kernel LLVM aborts on leaves its module behind.
    let dump = KernelDump::new(entrypoint);
    dump.write("nvptx.ll", || &ir);
    let ptx = compile(module, &eff_arch, ptx_version)?;
    dump.write("nvptx.ptx", || &ptx);

    Ok(NvptxModule {
        ptx: as_c_chars(&ptx),
        entrypoint: entrypoint.to_string(),
        ir,
        shared_memory_size: entry.shared_memory_size,
        io: entry.io,
    })
}

/// `ir` with the debug data of each compile unit as `DebugDirectivesOnly`: only the `.file` and
/// `.loc` directives, as `nvcc -lineinfo` gives. With line tables or full debug data, NVPTX writes
/// `.target <sm>, debug`, and the driver then compiles the kernel for a debugger, which changes the
/// optimization. The LLVM C API cannot create this kind, so the IR text is changed.
fn directives_only(ir: String, level: DebugInfo) -> String {
    let kind = match level {
        DebugInfo::None => return ir,
        DebugInfo::LineTables => "emissionKind: LineTablesOnly",
        DebugInfo::Full => "emissionKind: FullDebug",
    };
    ir.replace(kind, "emissionKind: DebugDirectivesOnly")
}

/// `ir` with a linkage name on each `DISubprogram` that has none: the inlined `#[cube]`
/// functions. NVPTX gives each `.loc` with `inlined_at` the linkage name of its function as
/// `function_name`, and `ptxas` copies it to the cubin. Without a linkage name, the name is
/// empty, and the inline frames in Nsight Compute have no name. The linkage name is the name,
/// because an inlined function has no symbol.
fn name_inlined_functions(ir: String) -> String {
    const NAME: &str = "DISubprogram(name: \"";
    if !ir.contains(NAME) {
        return ir;
    }
    let mut named = String::with_capacity(ir.len());
    for line in ir.split_inclusive('\n') {
        // The end of `name: "<name>"`, and the name.
        let name = line.find(NAME).and_then(|start| {
            let start = start + NAME.len();
            let len = line[start..].find('"')?;
            Some((start + len + 1, &line[start..start + len]))
        });
        match name {
            Some((end, name)) if !line.contains("linkageName:") => {
                named.push_str(&line[..end]);
                let _ = write!(named, ", linkageName: \"{name}\"");
                named.push_str(&line[end..]);
            }
            _ => named.push_str(line),
        }
    }
    named
}

/// Minimum PTX ISA version required by LLVM for `arch_version` (`sm_103a` / `sm_121a` require PTX 8.8).
fn min_ptx_for_arch(arch_version: u32) -> u32 {
    match arch_version {
        103 | 121 => 88,
        101 | 120 => 87,
        100 => 86,
        90..=99 => 80,
        _ => 63,
    }
}

/// When `ptx_version` is older than the minimum PTX version LLVM requires for a minor capability
/// (e.g. B300 `sm_103a` requires `ptx88`, while a CUDA 12.8 driver selects `ptx87`), step down to
/// the major family target (`sm_100a`) instead of triggering an LLVM `report_fatal_error` abort.
fn effective_arch(arch: &SmArch, ptx_version: Option<PtxVersion>) -> SmArch {
    if let Some(ptx) = ptx_version
        && ptx.number() < min_ptx_for_arch(arch.version())
        && arch.version() > 90
        && !arch.version().is_multiple_of(10)
    {
        return SmArch::new((arch.version() / 10) * 10, arch.tensor_cores());
    }
    *arch
}

/// Stamps the target and the entry point's calling convention and attributes on `module`.
fn finalize(
    module: &LlvmModule,
    entrypoint: &str,
    arch: &SmArch,
    entry: &NvptxEntry,
) -> Result<(), String> {
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
    let entry_fn = module.entry_point(entrypoint)?;
    entry_fn.set_calling_convention(PTX_KERNEL_CC);
    entry_fn.add_attributes(&attributes);
    annotate_buffer_params(
        &entry_fn,
        &entry.io,
        &entry.atomic_reads,
        entry.tensor_maps.bindings(),
        entry.metadata.count(),
    );
    if let MetadataParams::GridConstant { bytes, .. } = entry.metadata {
        mark_info_param_byval(&entry_fn, bytes);
    }
    entry.tensor_maps.mark(&entry_fn);
    Ok(())
}

/// Alignment required by the host metadata layout.
const INFO_PARAM_ALIGN: u32 = 8;

/// Passes the metadata by value: the last parameter, which the entry ABI lowering appended.
fn mark_info_param_byval(entry: &EntryFunction<'_>, bytes: usize) {
    let info = entry
        .last_param()
        .expect("the entry ABI lowering appended the info parameter");
    let byval = entry.add_param_byval(info, bytes as u64);
    let aligned = entry.add_param_attribute(info, "align", INFO_PARAM_ALIGN as u64);
    assert!(
        byval && aligned,
        "this LLVM has no `byval` or `align` attribute, so the grid-constant parameter cannot \
         be declared"
    );
}

/// `ptx_version` is `None` for LLVM's default, the oldest the architecture accepts.
fn compile(
    module: LlvmModule,
    arch: &SmArch,
    ptx_version: Option<PtxVersion>,
) -> Result<String, String> {
    init_nvptx();
    // Instruction selection reads `ALLOW_FTZ_ATOMICS`.
    let _options = read_llvm_options();
    let eff_arch = effective_arch(arch, ptx_version);

    let features = ptx_version
        .map(PtxVersion::target_feature)
        .unwrap_or_default();
    let machine = TargetMachine::new(&TargetSpec {
        triple: TRIPLE,
        cpu: &eff_arch.target_cpu(),
        features: &features,
        reloc: LLVMRelocMode::LLVMRelocDefault,
        code_model: LLVMCodeModel::LLVMCodeModelDefault,
    })?;
    machine.set_data_layout(&module);

    // SAFETY: the module is live for the call.
    unsafe { lower_to_device_libs(module.raw())? };
    module.run_passes(PASS_PIPELINE, &machine)?;
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
        let check = || {
            let ptx =
                compile_to_ptx(ir, &SmArch::new(60, false), PtxVersion::for_driver(12080)).unwrap();
            assert!(ptx.contains("atom.gpu.global.add.f32"), "{ptx}");
            assert!(!ptx.contains(".cas."), "no CAS loop:\n{ptx}");
        };
        check();
        // A link, even one that fails, sets LLVM's options back to their defaults.
        #[cfg(feature = "amdgpu")]
        {
            let _ = crate::amdgpu::lld::link_relocatable(b"", "k");
            check();
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

    #[test]
    fn blackwell_b200_and_b300_reach_tensor_core_instructions() {
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
        // On CUDA 12.9+ / 13.x (ptx88+), B300 targets native `.target sm_103a`
        for arch_ver in [100, 103] {
            let ptx_version = PtxVersion::for_driver(12090);
            let ptx = compile_to_ptx(ir, &SmArch::new(arch_ver, true), ptx_version).unwrap();
            assert!(
                ptx.contains(&format!(".target sm_{arch_ver}a")),
                "sm_{arch_ver}a target missing:\n{ptx}"
            );
            assert!(
                ptx.contains("ldmatrix.sync.aligned"),
                "sm_{arch_ver}a ldmatrix:\n{ptx}"
            );
            assert!(
                ptx.contains("mma.sync.aligned.m16n8k8"),
                "sm_{arch_ver}a mma:\n{ptx}"
            );
        }
        // On CUDA 12.8 (ptx87), B300 (103) gracefully targets `.target sm_100a` without aborting
        let ptx_12_8 =
            compile_to_ptx(ir, &SmArch::new(103, true), PtxVersion::for_driver(12080)).unwrap();
        assert!(ptx_12_8.contains(".target sm_100a"), "{ptx_12_8}");
        assert!(ptx_12_8.contains("mma.sync.aligned.m16n8k8"), "{ptx_12_8}");
    }

    fn compile_to_ptx(
        ir: &str,
        arch: &SmArch,
        ptx_version: Option<PtxVersion>,
    ) -> Result<String, String> {
        compile(LlvmModule::new(ir)?, arch, ptx_version)
    }
}
