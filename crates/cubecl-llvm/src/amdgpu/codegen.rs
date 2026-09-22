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
        AmdGpuModule,
        buffer_params::annotate_buffer_params,
        llvm_module::{EntryFunction, LlvmModule, TargetMachine, TargetSpec},
        math_library::redirect_intrinsics,
    },
};
use cubecl_core::ir::{amd::GfxArch, settings::Dim3};
use cubecl_environment::bytes::Bytes;
use llvm_sys::{
    LLVMAtomicRMWBinOp,
    target_machine::{LLVMCodeGenFileType, LLVMRelocMode},
};
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

/// Kernel entry point requirements.
pub struct AmdGpuEntry {
    /// Units per cube along each axis.
    pub cube_dim: Dim3,
    /// Shared memory required per launch, in bytes.
    pub shared_memory_size: usize,
    /// Buffer access modes in binding order.
    pub io: Vec<BufferIOAttr>,
}

pub fn emit_code_object(
    ctx: &Context,
    module: ModuleOp,
    entrypoint: &str,
    arch: &GfxArch,
    entry: AmdGpuEntry,
) -> Result<AmdGpuModule, String> {
    let llvm_ctx = LLVMContext::default();

    set_data_layout(ctx, module, DATA_LAYOUT.to_string());
    let converted =
        to_llvm_ir::convert_module(ctx, &llvm_ctx, module).map_err(|err| err.to_string())?;

    let module = LlvmModule::new(&converted.to_string())?;
    finalize(&module, entrypoint, arch, entry.cube_dim, &entry.io)?;
    let ir = module.print();
    let (object, asm) = compile(module, arch, Assembly::wanted())?;

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
        shared_memory_size: entry.shared_memory_size,
        io: entry.io,
    })
}

/// Whether a compile also emits the assembly, which only a debugging dump reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Assembly {
    Keep,
    Skip,
}

impl Assembly {
    /// `Keep` when `CUBECL_DEBUG_PLIRON` asks for the dumps.
    fn wanted() -> Self {
        match std::env::var_os("CUBECL_DEBUG_PLIRON").is_some() {
            true => Assembly::Keep,
            false => Assembly::Skip,
        }
    }
}

/// The metadata pointer `KernargArgs` appends after the buffers.
const METADATA_PARAMS: u32 = 1;

/// Stamps the target, the code object version and the entry point's calling convention and
/// attributes on `module`.
fn finalize(
    module: &LlvmModule,
    entrypoint: &str,
    arch: &GfxArch,
    cube_dim: Dim3,
    io: &[BufferIOAttr],
) -> Result<(), String> {
    let flat_work_group_size = format!("1,{}", cube_dim.num_elems());
    let mut attributes = vec![
        ("target-cpu", arch.name()),
        ("amdgpu-flat-work-group-size", &flat_work_group_size),
        // Every launch is a whole number of cubes.
        ("uniform-work-group-size", "true"),
    ];
    let features = features_for(arch);
    if !features.is_empty() {
        attributes.push(("target-features", features));
    }

    module.set_triple(TRIPLE);
    let entry = module.entry_point(entrypoint)?;
    entry.set_calling_convention(AMDGPU_KERNEL_CC);
    entry.add_attributes(&attributes);
    require_work_group_size(&entry, cube_dim);
    annotate_buffer_params(&entry, io, METADATA_PARAMS);
    mark_atomics_device_local(&entry);
    module.add_module_flag("amdhsa_code_object_version", CODE_OBJECT_VERSION);
    Ok(())
}

/// The cube dimensions are fixed when a kernel compiles, so the work-item ids are bounded by
/// them exactly: an axis of one unit is always zero, and the backend then neither unpacks its
/// id nor adds it into a position.
///
fn require_work_group_size(entry: &EntryFunction<'_>, cube_dim: Dim3) {
    entry.set_metadata(
        "reqd_work_group_size",
        &[cube_dim.x, cube_dim.y, cube_dim.z],
    );
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

fn mark_atomics_device_local(entry: &EntryFunction<'_>) {
    for inst in entry.instructions() {
        let Some(op) = inst.atomic_rmw_op() else {
            continue;
        };
        for kind in DEVICE_LOCAL_ATOMIC {
            entry.set_flag_metadata(&inst, kind);
        }
        if op == LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpFAdd {
            entry.set_flag_metadata(&inst, DENORMAL_AGNOSTIC_ATOMIC);
        }
    }
}

/// The relocatable object `module` compiles to, and its assembly when `want_asm` is set.
fn compile(
    module: LlvmModule,
    arch: &GfxArch,
    assembly: Assembly,
) -> Result<(Vec<u8>, Option<String>), String> {
    init_amdgpu();

    let features = CString::new(features_for(arch)).expect("static feature string");
    // AMD code objects require position-independent code.
    let machine = TargetMachine::new(&TargetSpec {
        triple: TRIPLE,
        cpu: arch.name(),
        features: &features,
        reloc: LLVMRelocMode::LLVMRelocPIC,
    })?;
    machine.set_data_layout(&module);

    // SAFETY: the module is live, stamped with the AMDGPU triple and layout.
    unsafe { lower_to_device_libs(module.raw(), arch)? };
    module.run_passes(PASS_PIPELINE, Some(&machine))?;

    // Emission consumes a module, so the assembly comes from a copy.
    let asm = if assembly == Assembly::Keep {
        let copy = LlvmModule::new(&module.print())?;
        let bytes = machine.emit(copy, LLVMCodeGenFileType::LLVMAssemblyFile)?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    } else {
        None
    };
    let object = machine.emit(module, LLVMCodeGenFileType::LLVMObjectFile)?;
    Ok((object, asm))
}

/// The object and assembly the finalized IR `ir` compiles to, for the tests that start from IR
/// rather than from a kernel, in this module and in `amdgpu::offline_tests`.
#[cfg(test)]
pub(crate) fn compile_to_object(
    ir: &str,
    arch: &GfxArch,
    assembly: Assembly,
) -> Result<(Vec<u8>, Option<String>), String> {
    compile(LlvmModule::new(ir)?, arch, assembly)
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
        let finalized =
            finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), Dim3::new_1d(64), &[]).unwrap();
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
        let finalized =
            finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), Dim3::new_1d(64), &[]).unwrap();
        let (object, asm) =
            compile_to_object(&finalized, &GfxArch::parse("gfx1201"), Assembly::Keep).unwrap();
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
            let finalized = finalize_ir(&ir, "k", &arch, Dim3::new_1d(32), &[]).unwrap();
            let (object, asm) = compile_to_object(&finalized, &arch, Assembly::Keep).unwrap();
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
        let finalized =
            finalize_ir(ir, "k", &arch, Dim3::new_1d(64), &[BufferIOAttr::WriteOnly]).unwrap();
        assert!(finalized.contains("noalias"), "{finalized}");
        assert!(
            finalized.contains("readonly"),
            "the metadata is read-only:\n{finalized}"
        );

        let (_, asm) = compile_to_object(&finalized, &arch, Assembly::Keep).unwrap();
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
            let finalized = finalize_ir(ir, "k", &arch, Dim3::new_1d(64), &[]).unwrap();
            let (_, asm) = compile_to_object(&finalized, &arch, Assembly::Keep).unwrap();
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
        let finalized =
            finalize_ir(ir, "k", &GfxArch::parse("gfx1201"), Dim3::new_1d(64), &[]).unwrap();
        let (object, asm) =
            compile_to_object(&finalized, &GfxArch::parse("gfx1201"), Assembly::Keep).unwrap();
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

    fn finalize_ir(
        ir: &str,
        entrypoint: &str,
        arch: &GfxArch,
        cube_dim: Dim3,
        io: &[BufferIOAttr],
    ) -> Result<String, String> {
        let module = LlvmModule::new(ir)?;
        finalize(&module, entrypoint, arch, cube_dim, io)?;
        Ok(module.print())
    }
}
