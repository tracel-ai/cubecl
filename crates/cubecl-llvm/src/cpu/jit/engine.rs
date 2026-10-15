use super::{
    data::PlironData,
    lljit::Jit,
    symbols::{JitSymbols, write_perf_map},
};
use crate::{
    cpu::shared_memory::SharedMemories,
    prelude::{Context, ModuleOp},
    shared::{
        debug_info::{check_debug_info, convert_module},
        llvm_module::{LlvmModule, TargetMachine},
    },
};
use cubecl_core::{codegen::KernelDump, ir::settings::DebugInfo};
use cubecl_environment::{backtrace::BackTrace, bytes::Bytes};
use cubecl_runtime::{compiler::CompilationError, kernel::BufferIOAttr};
use llvm_sys::target_machine::LLVMCodeGenFileType;
use pliron_llvm::llvm_sys::core::LLVMContext;
use std::{
    ffi::{CStr, c_void},
    fmt::Display,
    sync::{Arc, Mutex, Once},
};

/// Kernel ABI: buffer pointers, cube count x/y/z, unit position x/y/z,
/// barrier state, metadata.
type KernelFn = extern "C" fn(*mut *mut c_void, u32, u32, u32, u32, u32, u32, *mut u32, *mut u64);

/// Resources and scheduling required for a launch.
#[derive(Clone, Debug, Default)]
pub struct KernelRequirements {
    /// Cube barriers require a separate thread for each unit.
    pub needs_parallelism: bool,
    /// Shared memory required for a launch.
    pub shared_memories: SharedMemories,
}

/// A compiled kernel and its owning JIT.
#[repr(C)]
struct JitKernel {
    func: KernelFn,
    requirements: KernelRequirements,
    /// Buffer access modes in binding order.
    io: Vec<BufferIOAttr>,
    object_bytes: Mutex<Option<Bytes>>,
    _jit: Jit,
}

/// SAFETY: Compiled code is immutable and its JIT owns the context.
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

#[derive(Clone)]
pub struct PlironEngine(Arc<JitKernel>);

static INIT_NATIVE: Once = Once::new();

pub(crate) fn init_native() {
    INIT_NATIVE.call_once(|| {
        pliron_llvm::llvm_sys::target::initialize_native()
            .expect("failed to initialize native target");
    });
}

impl PlironEngine {
    pub fn compile(
        ctx: &Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
        keep_object: bool,
    ) -> pliron::result::Result<Self> {
        Self::compile_with_debug_info(
            ctx,
            module,
            kernel_name,
            requirements,
            io,
            DebugInfo::None,
            keep_object,
        )
    }

    /// [`compile`](Self::compile), for a kernel that carries `debug_info`. With debug data, the
    /// profiler symbol files that the environment asks for are written. With `keep_object`, the
    /// engine keeps the object code for [`take_object_bytes`](Self::take_object_bytes).
    ///
    /// # Errors
    /// When the module does not convert to LLVM IR.
    ///
    /// # Panics
    /// When LLVM cannot create the JIT or the target machine, or cannot optimize, emit or link
    /// the kernel.
    pub fn compile_with_debug_info(
        ctx: &Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
        debug_info: DebugInfo,
        keep_object: bool,
    ) -> pliron::result::Result<Self> {
        init_native();

        let llvm_module = to_llvm_module(ctx, module, kernel_name, debug_info)?;
        let dump = KernelDump::new(kernel_name);
        dump.write("llvm.ll", || llvm_module.print());

        let symbols = match debug_info {
            DebugInfo::None => JitSymbols::default(),
            _ => JitSymbols::from_env(),
        };
        #[cfg(cubecl_frame_pointers)]
        llvm_module.add_function_attribute("frame-pointer", "all");
        // The perf support plugin copies the `.eh_frame` of each kernel into the jitdump. Then
        // `perf --call-graph dwarf` can unwind through the kernel. `2` is `uwtable(async)`, as rustc
        // gives the host code.
        #[cfg(feature = "jitdump")]
        if symbols.jitdump {
            llvm_module.add_function_enum_attribute("uwtable", 2);
        }
        let jit = Jit::new(symbols, debug_info != DebugInfo::None).expect("failed to create LLJIT");
        let machine = TargetMachine::native(&llvm_module, jit.code_model())
            .unwrap_or_else(|err| panic!("failed to create native target machine: {err}"));
        llvm_module
            .run_passes(PASS_PIPELINE, &machine)
            .unwrap_or_else(|err| panic!("LLVM optimization failed for '{kernel_name}': {err}"));
        dump.write("llvm.opt.ll", || llvm_module.print());

        let object = machine
            .emit(llvm_module, LLVMCodeGenFileType::LLVMObjectFile)
            .unwrap_or_else(|err| panic!("LLVM code generation failed for '{kernel_name}': {err}"));
        dump.write("kernel.o", || &object);

        let engine =
            Self::load(jit, &object, kernel_name, requirements, io).unwrap_or_else(|err| {
                panic!("failed to load compiled object for '{kernel_name}': {err}")
            });
        if keep_object {
            *engine.0.object_bytes.lock().unwrap() = Some(Bytes::from_bytes_vec(object));
        }
        Ok(engine)
    }

    /// Load precompiled object bytes directly into the JIT linking layer.
    ///
    /// # Errors
    /// When the JIT cannot link the object, or the object does not define `entrypoint`.
    pub fn from_object(
        bytes: &[u8],
        entrypoint: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
    ) -> Result<Self, CompilationError> {
        init_native();

        let generic = |reason| CompilationError::Generic {
            reason,
            backtrace: BackTrace::capture(),
        };
        let jit = Jit::new(JitSymbols::default(), false)
            .map_err(|err| generic(format!("failed to create the JIT: {err}")))?;
        Self::load(jit, bytes, entrypoint, requirements, io).map_err(generic)
    }

    /// Links the object code `bytes` into `jit`, and takes the kernel `entrypoint` from it.
    fn load(
        jit: Jit,
        bytes: &[u8],
        entrypoint: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
    ) -> Result<Self, String> {
        jit.add_object_file(bytes, entrypoint)
            .map_err(|err| format!("failed to add object file to JIT: {err}"))?;
        let addr = jit
            .lookup(entrypoint)
            .map_err(|err| format!("kernel symbol '{entrypoint}' not found: {err}"))?;
        if let Some(size) = jit.symbol_size(entrypoint) {
            write_perf_map(addr, size, entrypoint);
        }
        // SAFETY: The generated entry point matches `KernelFn`.
        let func: KernelFn = unsafe { std::mem::transmute::<u64, KernelFn>(addr) };

        Ok(PlironEngine(Arc::new(JitKernel {
            func,
            requirements,
            io,
            object_bytes: Mutex::new(None),
            _jit: jit,
        })))
    }

    /// Takes the precompiled object bytes by value, leaving None in the engine.
    pub fn take_object_bytes(&self) -> Option<Bytes> {
        self.0.object_bytes.lock().unwrap().take()
    }

    pub fn requirements(&self) -> &KernelRequirements {
        &self.0.requirements
    }

    /// Buffer access modes in binding order.
    pub fn buffer_io(&self) -> &[BufferIOAttr] {
        &self.0.io
    }

    pub fn run_kernel(&self, data: &mut PlironData) {
        let b = data.builtins;
        let buffer_ptrs = data.shared.buffer_ptrs.as_ptr() as *mut *mut c_void;
        let metadata = data.shared.metadata.as_ptr() as *mut u64;
        let sync_cube_state = data.shared.sync_cube_state.as_ptr() as *mut u32;
        (self.0.func)(
            buffer_ptrs,
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
            sync_cube_state,
            metadata,
        );
    }
}

impl Display for PlironEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pliron JIT engine")
    }
}

/// Converts `module` to an LLVM module, with the DWARF of `debug_info`.
pub(crate) fn to_llvm_module(
    ctx: &Context,
    module: ModuleOp,
    kernel_name: &str,
    debug_info: DebugInfo,
) -> pliron::result::Result<LlvmModule> {
    let llvm_ctx = LLVMContext::default();
    let llvm_module = convert_module(ctx, &llvm_ctx, module, debug_info, true)?;
    let llvm_module = LlvmModule::new(&llvm_module.to_string())
        .unwrap_or_else(|err| panic!("LLVM IR does not parse for '{kernel_name}': {err}"));
    check_debug_info(&llvm_module, kernel_name, debug_info);
    Ok(llvm_module)
}

/// Optimization pipeline for JIT compilation.
const PASS_PIPELINE: &CStr = c"default<O3>";
