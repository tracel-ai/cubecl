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
        llvm_module::LlvmModule,
    },
};
use cubecl_core::{codegen::KernelDump, ir::settings::DebugInfo};
use cubecl_runtime::kernel::BufferIOAttr;
use pliron_llvm::llvm_sys::{core::LLVMContext, target::initialize_native};
use std::{
    ffi::{CStr, c_void},
    fmt::Display,
    sync::{Arc, Once},
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
    _jit: Jit,
}

/// SAFETY: Compiled code is immutable and its JIT owns the context.
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

#[derive(Clone)]
pub struct PlironEngine(Arc<JitKernel>);

static INIT_NATIVE: Once = Once::new();

impl PlironEngine {
    pub fn compile(
        ctx: &Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
    ) -> pliron::result::Result<Self> {
        Self::compile_with_debug_info(ctx, module, kernel_name, requirements, io, DebugInfo::None)
    }

    /// [`compile`](Self::compile), for a kernel that carries `debug_info`. With debug data, the JIT
    /// registers the kernel with gdb, and the profiler symbol files that the environment asks for
    /// are written.
    ///
    /// # Errors
    /// The conversion error, when `module` does not convert to LLVM IR.
    ///
    /// # Panics
    /// When LLVM rejects the IR or cannot compile it: the lowering has a bug.
    pub fn compile_with_debug_info(
        ctx: &Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
        io: Vec<BufferIOAttr>,
        debug_info: DebugInfo,
    ) -> pliron::result::Result<Self> {
        INIT_NATIVE.call_once(|| {
            initialize_native().expect("failed to initialize native target");
        });

        let llvm_module = to_llvm_module(ctx, module, kernel_name, debug_info)?;
        let dump = KernelDump::new(kernel_name);
        dump.write("llvm.ll", || llvm_module.print());

        let symbols = match debug_info {
            DebugInfo::None => JitSymbols::default(),
            _ => JitSymbols::from_env(),
        };

        llvm_module
            .run_passes(PASS_PIPELINE, None)
            .unwrap_or_else(|err| panic!("LLVM optimization failed for '{kernel_name}': {err}"));
        dump.write("llvm.opt.ll", || llvm_module.print());

        let jit = Jit::new(symbols, debug_info != DebugInfo::None).expect("failed to create LLJIT");
        jit.add_module(llvm_module)
            .expect("failed to add module to JIT");
        let addr = jit
            .lookup(kernel_name)
            .unwrap_or_else(|err| panic!("kernel symbol '{kernel_name}' not found: {err}"));
        if let Some(size) = jit.symbol_size(kernel_name) {
            write_perf_map(addr, size, kernel_name);
        }
        // SAFETY: The generated entry point matches `KernelFn`.
        let func: KernelFn = unsafe { std::mem::transmute::<u64, KernelFn>(addr) };

        Ok(PlironEngine(Arc::new(JitKernel {
            func,
            requirements,
            io,
            _jit: jit,
        })))
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
