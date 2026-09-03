//! Compiling the LLVM dialect to PTX.
//!
//! Shorter than the AMDGPU path by the linking that path needs and this one does not. The
//! NVPTX backend emits PTX assembly rather than an object file, and PTX is what the CUDA
//! driver already takes: `cuModuleLoadData` JITs it exactly as it does the PTX NVRTC hands
//! back today, so there is no LLD step and nothing to turn into a shared object. What is left
//! is the same shape — stamp the module, pull in the device library the math needs, run the
//! pipeline, emit — with `LLVMAssemblyFile` as the output kind.

use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron_llvm::llvm_sys::core::LLVMContext;
use pliron_llvm::to_llvm_ir;
use std::ffi::{CStr, CString};
use std::sync::Once;

use crate::nvptx::libdevice::{Libdevice, link_libdevice};
use crate::nvptx::printf::lower_printf_to_vprintf;
use crate::shared::NvptxModule;
use crate::shared::math_library::redirect_intrinsics;
use cubecl_core::ir::attributes::BufferIOAttr;
use cubecl_core::ir::nvidia::SmArch;

/// The NVPTX target triple; the specific device is the `-mcpu`, not the triple.
const TRIPLE: &CStr = c"nvptx64-nvidia-cuda";

/// LLVM's calling convention number for `ptx_kernel`
/// (`llvm::CallingConv::PTX_Kernel`). What marks a function as a grid entry point: since
/// LLVM 21 the `nvvm.annotations` metadata that used to say so is gone, and this is the only
/// way left to say it.
const PTX_KERNEL_CC: u32 = 71;

/// No subtarget features are asked for, which is deliberate: the one that would go here is the
/// PTX ISA version, and LLVM already picks the lowest that supports the `-mcpu` -- 7.1 for
/// `sm_86`. That is both the most compatible answer, since the driver rejects a `.version`
/// above what it knows, and the only safe one: an architecture needs a minimum ISA version to
/// be nameable at all, and pinning one below it is a hard `LLVM ERROR` rather than a
/// diagnostic. `sm_90a` needs 8.0, and the Blackwell parts more again.
const NO_FEATURES: &CStr = c"";

/// The pipeline run before machine-code emission. The same one the AMDGPU path runs, handed
/// the target machine, which is what makes it target-aware.
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

/// Lowers `module` to LLVM IR and compiles it to PTX for `arch`.
pub fn emit_ptx(
    ctx: &Context,
    module: ModuleOp,
    entrypoint: &str,
    arch: &SmArch,
    cube_dim: u32,
    shared_memory_size: usize,
    io: Vec<BufferIOAttr>,
) -> Result<NvptxModule, String> {
    let llvm_ctx = LLVMContext::default();

    // No data layout is set here, where the AMDGPU path has to name address space 5 for its
    // allocas: NVPTX puts them in the generic space, so the layout the target machine reports
    // below is the whole of it.
    let llvm_module =
        to_llvm_ir::convert_module(ctx, &llvm_ctx, module).map_err(|err| err.to_string())?;

    let ir = finalize_ir(&llvm_module.to_string(), entrypoint, arch, cube_dim, &io)?;
    let ptx = compile_to_ptx(&ir, arch)?;

    #[cfg(feature = "pliron-dump")]
    if let Some(dir) = crate::cpu::jit::engine::ir_dump_path(entrypoint) {
        let _ = std::fs::write(dir.join("nvptx.ll"), &ir);
        let _ = std::fs::write(dir.join("nvptx.ptx"), &ptx);
    }

    Ok(NvptxModule {
        ptx: as_c_chars(&ptx),
        entrypoint: entrypoint.to_string(),
        ir,
        shared_memory_size,
        io,
    })
}

/// Stamps `ir` with what the NVPTX backend keys off: the triple, the `ptx_kernel` calling
/// convention on `entrypoint`, and the launch bounds.
fn finalize_ir(
    ir: &str,
    entrypoint: &str,
    arch: &SmArch,
    cube_dim: u32,
    io: &[BufferIOAttr],
) -> Result<String, String> {
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMContextDispose, LLVMCreateStringAttribute, LLVMDisposeMessage,
        LLVMDisposeModule, LLVMGetNamedFunction, LLVMPrintModuleToString, LLVMSetFunctionCallConv,
        LLVMSetTarget,
    };

    // Built before the module is parsed so this error path has nothing to dispose.
    let name = CString::new(entrypoint)
        .map_err(|_| format!("kernel name '{entrypoint}' contains a NUL"))?;

    let target_cpu = arch.target_cpu();
    // `maxntid` is what `__launch_bounds__` sets: it caps the registers the kernel may use so
    // a cube of this size can be resident, and a cube launched larger than it is rejected.
    // One number rather than three because the launch flattens the cube the same way.
    let max_threads = cube_dim.to_string();
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

        annotate_buffer_params(ctx, func, io);

        let c_ir = LLVMPrintModuleToString(module);
        let finalized = CStr::from_ptr(c_ir).to_string_lossy().into_owned();
        LLVMDisposeMessage(c_ir);
        LLVMDisposeModule(module);
        LLVMContextDispose(ctx);
        Ok(finalized)
    }
}

/// Tells the backend what the kernel's pointer parameters are, which is the other half of what
/// puts a load in the read-only cache.
///
/// `NVPTXTagInvariantLoads` marks a load invariant -- and only an invariant load becomes
/// `ld.global.nc` -- when every object behind its pointer is a kernel parameter that is both
/// `readonly` and `noalias`. The address space comes from the entry ABI; these are the
/// attributes, and they are exactly what the C++ backend asserts by declaring every binding
/// `const __restrict__`.
///
/// `noalias` goes on every buffer for the same reason `__restrict__` does: two bindings of one
/// launch are distinct allocations, which is a contract `CubeCL` already relies on everywhere
/// else. `readonly` goes only where the compiler proved it, which is what
/// `AnnotateGlobalVisibilityPass` computed and the launch path already trusts to decide which
/// buffers a failed kernel taints.
///
/// # Safety
/// `func` must be a live function in `ctx` whose parameters are the buffers in binding order
/// followed by the metadata pointer.
unsafe fn annotate_buffer_params(
    ctx: llvm_sys::prelude::LLVMContextRef,
    func: llvm_sys::prelude::LLVMValueRef,
    io: &[BufferIOAttr],
) {
    use llvm_sys::LLVMTypeKind;
    use llvm_sys::core::{
        LLVMAddAttributeAtIndex, LLVMCountParams, LLVMCreateEnumAttribute,
        LLVMGetEnumAttributeKindForName, LLVMGetParam, LLVMGetTypeKind, LLVMTypeOf,
    };

    unsafe {
        let enum_attr = |index: u32, name: &str| {
            let kind = LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
            // Zero is "no such attribute": building one from it crashes rather than
            // diagnosing, so a name this LLVM does not know is skipped instead.
            if kind == 0 {
                return;
            }
            let attribute = LLVMCreateEnumAttribute(ctx, kind, 0);
            LLVMAddAttributeAtIndex(func, index, attribute);
        };

        // Whether `readonly` is safe to state at all here; see `reads_atomically`.
        let may_say_readonly = !reads_atomically(func);

        let params = LLVMCountParams(func);
        // The metadata pointer is the last parameter, past the buffers, and the kernel only
        // ever reads it.
        let info = params.saturating_sub(1);
        for param in 0..params {
            // Both attributes are only meaningful on a pointer, and applying one to anything
            // else is rejected by the verifier rather than ignored.
            if LLVMGetTypeKind(LLVMTypeOf(LLVMGetParam(func, param)))
                != LLVMTypeKind::LLVMPointerTypeKind
            {
                continue;
            }

            // Parameter attributes are indexed from one; zero is the return value.
            let index = param + 1;
            enum_attr(index, "noalias");

            let read_only = param == info
                || io
                    .get(param as usize)
                    .is_some_and(|attr| *attr == BufferIOAttr::ReadOnly);
            if read_only && may_say_readonly {
                enum_attr(index, "readonly");
            }
        }
    }
}

/// Whether `func` loads atomically, which is what makes `readonly` unsafe to state on any of
/// its parameters.
///
/// `NVPTXTagInvariantLoads` matches every `LoadInst` whose pointer comes from a `readonly
/// noalias` kernel parameter in the global space, with no guard against the load being atomic,
/// and `ISel` then sends it to `tryLDG` -- which emits `ld.global.nc`, an instruction that has no
/// atomic form. In LLVM 23 that is a segfault in the backend rather than a diagnostic.
///
/// The attribute is what creates the precondition, so the attribute is what is withheld. It is
/// withheld for the whole function rather than for the one parameter because the pass looks
/// through the pointer's whole use chain to find the argument behind it, and reproducing that
/// reachability here to be precise would be a worse trade than losing the read-only cache in
/// the kernels that use atomics at all.
///
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

/// Compiles `ir` to PTX for `arch`.
fn compile_to_ptx(ir: &str, arch: &SmArch) -> Result<String, String> {
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
    let features = NO_FEATURES;

    unsafe {
        let mut target = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        if LLVMGetTargetFromTriple(TRIPLE.as_ptr(), &mut target, &mut error) != 0 {
            let message = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            return Err(message);
        }

        // PTX has no relocations to speak of -- the driver's JIT resolves everything when it
        // loads the module -- so the reloc model is the default rather than the PIC the
        // AMDGPU path needs to get a shared object out of LLD.
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

        // Layout from the target machine, so the two can never drift apart.
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

/// Rewrites what the NVPTX backend cannot emit as it stands: the transcendentals into
/// `libdevice` calls, with the library linked in behind them, and the variadic `printf` into
/// the `vprintf` the device actually has.
///
/// Before the pipeline rather than after, so the inliner sees the library bodies and the
/// optimizer works on what they expand to -- which is the whole reason the toolkit ships
/// `libdevice` as bitcode rather than as a linked library.
///
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

/// One `LLVMTargetMachineEmitToMemoryBuffer` call, copied out and disposed.
///
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

/// Parses textual `ir` into a fresh context, returning both so the caller can dispose them.
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

/// PTX as the NUL-terminated `c_char` buffer `cuModuleLoadData` reads and the compilation cache
/// stores, which is the shape NVRTC hands its own PTX back in.
fn as_c_chars(ptx: &str) -> Vec<std::ffi::c_char> {
    let mut bytes: Vec<std::ffi::c_char> =
        ptx.bytes().map(|byte| byte as std::ffi::c_char).collect();
    // The driver reads to the terminator rather than being given a length, and LLVM's buffer
    // does not carry one.
    bytes.push(0);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The finalized module carries everything the NVPTX backend needs: without the calling
    /// convention the function is a device function and no entry point at all, so the module
    /// would load and `cuModuleGetFunction` would then not find it.
    #[test]
    fn finalize_sets_triple_callconv_and_arch() {
        let ir = r#"
define void @k(ptr %out) {
entry:
  store i32 7, ptr %out, align 4
  ret void
}
"#;
        let finalized = finalize_ir(ir, "k", &SmArch::new(86, true), 256, &[]).unwrap();
        assert!(
            finalized.contains(r#"target triple = "nvptx64-nvidia-cuda""#),
            "{finalized}"
        );
        assert!(finalized.contains("ptx_kernel"), "{finalized}");
        assert!(finalized.contains(r#""target-cpu"="sm_86""#), "{finalized}");
        assert!(finalized.contains(r#""nvvm.maxntid"="256""#), "{finalized}");
    }

    /// A kernel reaches PTX with the entry point visible to `cuModuleGetFunction`, and the
    /// `.target` the device was named for.
    #[test]
    fn emits_ptx_naming_its_entry_point() {
        let ir = r#"
define void @k(ptr %out) {
entry:
  store i32 7, ptr %out, align 4
  ret void
}
"#;
        let arch = SmArch::new(86, true);
        let finalized = finalize_ir(ir, "k", &arch, 256, &[]).unwrap();
        let ptx = compile_to_ptx(&finalized, &arch).unwrap();

        assert!(ptx.contains(".target sm_86"), "{ptx}");
        assert!(ptx.contains(".visible .entry k("), "{ptx}");
        assert!(ptx.contains(".maxntid 256"), "{ptx}");
    }

    /// The shared memory block reaches the PTX as `.extern .shared`, which is what makes it
    /// dynamic: the module reserves none of its own and the launch gives it a block through
    /// `sharedMemBytes`. The generic pointers the rest of the pipeline works with are inferred
    /// away, and the barrier is the hardware's own.
    #[test]
    fn shared_memory_becomes_dynamic_shared() {
        let ir = r#"
@cube_shared = external addrspace(3) global [0 x i8], align 16
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32)
define void @k(ptr %out, i32 %tid) {
entry:
  %slice = getelementptr i8, ptr addrspace(3) @cube_shared, i32 64
  %flat = addrspacecast ptr addrspace(3) %slice to ptr
  %idx = getelementptr float, ptr %flat, i32 %tid
  store float 1.0, ptr %idx, align 4
  fence syncscope("block") release
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)
  fence syncscope("block") acquire
  %v = load float, ptr %idx, align 4
  store float %v, ptr %out
  ret void
}
"#;
        let arch = SmArch::new(86, true);
        let finalized = finalize_ir(ir, "k", &arch, 256, &[]).unwrap();
        let ptx = compile_to_ptx(&finalized, &arch).unwrap();

        assert!(
            ptx.contains(".extern .shared .align 16 .b8 cube_shared[]"),
            "the block should be dynamically sized:\n{ptx}"
        );
        assert!(
            ptx.contains("st.shared") && ptx.contains("ld.shared"),
            "the accesses should reach shared memory:\n{ptx}"
        );
        assert!(
            !ptx.contains("st.f32") || !ptx.contains("ld.f32"),
            "the generic pointers should be inferred away:\n{ptx}"
        );
        assert!(ptx.contains("bar.sync"), "the cube barrier:\n{ptx}");
    }

    /// The cross-lane instructions the plane lowering is built on reach the PTX. Getting an
    /// intrinsic name wrong fails to select rather than computing the wrong answer, so this
    /// pins the ones every plane operation goes through.
    #[test]
    fn the_plane_primitives_reach_the_ptx() {
        let ir = r#"
declare i32 @llvm.nvvm.shfl.sync.idx.i32(i32, i32, i32, i32)
declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1)
declare i32 @llvm.nvvm.read.ptx.sreg.laneid()
define void @k(ptr %out, i32 %v) {
entry:
  %lane = call i32 @llvm.nvvm.read.ptx.sreg.laneid()
  %s = call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %v, i32 %lane, i32 31)
  %p = icmp sgt i32 %s, 0
  %b = call i32 @llvm.nvvm.vote.ballot.sync(i32 -1, i1 %p)
  store i32 %b, ptr %out, align 4
  ret void
}
"#;
        let arch = SmArch::new(86, true);
        let finalized = finalize_ir(ir, "k", &arch, 32, &[]).unwrap();
        let ptx = compile_to_ptx(&finalized, &arch).unwrap();

        assert!(ptx.contains("%laneid"), "{ptx}");
        assert!(ptx.contains("shfl.sync.idx"), "{ptx}");
        assert!(ptx.contains("vote.sync.ballot"), "{ptx}");
    }

    /// The matrix instructions reach the PTX, in the shape the lowering names them.
    ///
    /// Getting an intrinsic's name or its register count wrong fails to select rather than
    /// computing a wrong answer, and needs no device to catch, so the three the cooperative
    /// path goes through are pinned here: the load, the multiply and the store. The register
    /// counts are the ones `nvptx::matrix` builds its fragments from.
    #[test]
    fn the_matrix_instructions_reach_the_ptx() {
        let list = |ty: &str| [ty; 8].join(", ");
        let (half8, float8) = (list("<2 x half>"), list("float"));
        let named = |ty: &str, prefix: &str| {
            (0..8)
                .map(|i| format!("{ty} %{prefix}{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let extract = |ty: &str, from: &str, prefix: &str| {
            (0..8)
                .map(|i| format!("  %{prefix}{i} = extractvalue {{{ty}}} %{from}, {i}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let zeros = list("float 0.0");

        let ir = format!(
            "declare {{{half8}}} @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr, i32)\n\
             declare {{{float8}}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32({half8}, \
             {half8}, {float8})\n\
             declare void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr, {float8}, \
             i32)\n\
             define void @k(ptr %src, ptr %dst) {{\n\
             entry:\n\
               %a = call {{{half8}}} \
             @llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0(ptr %src, i32 16)\n\
             {a_regs}\n\
               %d = call {{{float8}}} @llvm.nvvm.wmma.m16n16k16.mma.row.col.f32.f32({a_args}, \
             {a_args}, {zeros})\n\
             {d_regs}\n\
               call void @llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0(ptr %dst, \
             {d_args}, i32 16)\n\
               ret void\n\
             }}\n",
            a_regs = extract(&half8, "a", "a"),
            d_regs = extract(&float8, "d", "d"),
            a_args = named("<2 x half>", "a"),
            d_args = named("float", "d"),
        );

        let arch = SmArch::new(86, true);
        let finalized = finalize_ir(&ir, "k", &arch, 32, &[]).unwrap();
        let ptx = compile_to_ptx(&finalized, &arch).unwrap();

        assert!(
            ptx.contains("wmma.load.a.sync.aligned.row.m16n16k16.f16"),
            "{ptx}"
        );
        assert!(
            ptx.contains("wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32"),
            "{ptx}"
        );
        assert!(
            ptx.contains("wmma.store.d.sync.aligned.row.m16n16k16.f32"),
            "{ptx}"
        );
    }

    /// The manual `mma.sync` family reaches the PTX: the multiply itself, and the two tile
    /// moves that feed it.
    ///
    /// `test_cmma_manual` covers the multiply end to end on a device, but nothing in the suite
    /// reaches `ldmatrix` without a barrier to stage through, and `stmatrix` is `sm_90` and up
    /// where this machine is `sm_86`. So the names are pinned here, where no device is needed:
    /// a wrong one fails to select rather than computing a wrong answer.
    #[test]
    fn the_manual_matrix_instructions_reach_the_ptx() {
        let half2 = |n: usize| vec!["<2 x half>"; n].join(", ");
        let float4 = ["float"; 4].join(", ");
        let named = |ty: &str, prefix: &str, n: usize| {
            (0..n)
                .map(|i| format!("{ty} %{prefix}{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        };

        // `mma.sync` at the shape the manual API is built around, on `sm_86`.
        let ir = format!(
            "declare {{{float4}}} @llvm.nvvm.mma.m16n8k16.row.col.f32.f32({a}, {b}, {float4})\n\
             define void @k(ptr %dst, {a_named}, {b_named}) {{\n\
             entry:\n\
               %d = call {{{float4}}} @llvm.nvvm.mma.m16n8k16.row.col.f32.f32({a_args}, \
             {b_args}, float 0.0, float 0.0, float 0.0, float 0.0)\n\
               %d0 = extractvalue {{{float4}}} %d, 0\n\
               store float %d0, ptr %dst\n\
               ret void\n\
             }}\n",
            a = half2(4),
            b = half2(2),
            a_named = named("<2 x half>", "a", 4),
            b_named = named("<2 x half>", "b", 2),
            a_args = named("<2 x half>", "a", 4),
            b_args = named("<2 x half>", "b", 2),
        );
        let arch = SmArch::new(86, true);
        let ptx = compile_to_ptx(&finalize_ir(&ir, "k", &arch, 32, &[]).unwrap(), &arch).unwrap();
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "{ptx}"
        );

        // `ldmatrix` reads its tiles out of shared memory, which is why the pointer is cast
        // into address space 3 before the call.
        let ir = "\
declare {i32, i32} @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3))
define void @k(ptr addrspace(3) %src, ptr %dst) {
entry:
  %r = call {i32, i32} @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3) %src)
  %r0 = extractvalue {i32, i32} %r, 0
  store i32 %r0, ptr %dst
  ret void
}
";
        let ptx = compile_to_ptx(&finalize_ir(ir, "k", &arch, 32, &[]).unwrap(), &arch).unwrap();
        assert!(
            ptx.contains("ldmatrix.sync.aligned.m8n8.x2.shared"),
            "{ptx}"
        );

        // `stmatrix` only exists from `sm_90`, which is also the only architecture the runtime
        // advertises it on.
        let ir = "\
declare void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3), i32, i32)
define void @k(ptr addrspace(3) %dst, i32 %a, i32 %b) {
entry:
  call void @llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3(ptr addrspace(3) %dst, i32 %a, i32 %b)
  ret void
}
";
        let hopper = SmArch::new(90, true);
        let ptx =
            compile_to_ptx(&finalize_ir(ir, "k", &hopper, 32, &[]).unwrap(), &hopper).unwrap();
        assert!(
            ptx.contains("stmatrix.sync.aligned.m8n8.x2.shared"),
            "{ptx}"
        );
    }
}
