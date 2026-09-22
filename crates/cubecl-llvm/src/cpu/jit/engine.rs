use super::data::PlironData;
use crate::{
    cpu::shared_memory::SharedMemories,
    prelude::{Context, ModuleOp},
    shared::llvm_module::LlvmModule,
};
use cubecl_runtime::kernel::BufferIOAttr;
use pliron_llvm::{
    llvm_sys::{
        core::{LLVMContext, LLVMMemoryBuffer, LLVMModule},
        lljit::LLVMLLJIT,
        target::initialize_native,
    },
    to_llvm_ir,
};
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
    _lljit: LLVMLLJIT,
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
        INIT_NATIVE.call_once(|| {
            initialize_native().expect("failed to initialize native target");
        });

        let llvm_ctx = LLVMContext::default();
        let llvm_module = to_llvm_ir::convert_module(ctx, &llvm_ctx, module)?;
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.ll"), llvm_module.to_string());
        }

        let llvm_module = optimize(llvm_module, &llvm_ctx, kernel_name)
            .unwrap_or_else(|err| panic!("LLVM optimization failed for '{kernel_name}': {err}"));
        #[cfg(feature = "pliron-dump")]
        if let Some(dir) = ir_dump_path(kernel_name) {
            let _ = std::fs::write(dir.join("llvm.opt.ll"), llvm_module.to_string());
        }

        let lljit = LLVMLLJIT::new_with_default_builder().expect("failed to create LLJIT");
        lljit
            .add_module(llvm_ctx, llvm_module)
            .expect("failed to add module to JIT");
        let addr = lljit
            .lookup_symbol(kernel_name)
            .unwrap_or_else(|err| panic!("kernel symbol '{kernel_name}' not found: {err}"));
        // SAFETY: The generated entry point matches `KernelFn`.
        let func: KernelFn = unsafe { std::mem::transmute::<u64, KernelFn>(addr) };

        Ok(PlironEngine(Arc::new(JitKernel {
            func,
            requirements,
            io,
            _lljit: lljit,
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

#[cfg(feature = "pliron-dump")]
/// IR dump directory, enabled by `CUBECL_DEBUG_PLIRON`.
pub(crate) fn ir_dump_path(kernel_name: &str) -> Option<std::path::PathBuf> {
    let dir = std::env::var("CUBECL_DEBUG_PLIRON").ok()?;
    let path = std::path::Path::new(&dir).join(kernel_name);
    std::fs::create_dir_all(&path).ok()?;
    Some(path)
}

/// Optimization pipeline for JIT compilation.
const PASS_PIPELINE: &CStr = c"default<O3>";

fn optimize(
    module: LLVMModule,
    llvm_ctx: &LLVMContext,
    kernel_name: &str,
) -> Result<LLVMModule, String> {
    let optimized = run_pipeline(&module.to_string())?;
    drop(module);
    LLVMModule::from_ir_in_memory_buffer(
        llvm_ctx,
        LLVMMemoryBuffer::from_str(&optimized, kernel_name),
    )
}

fn run_pipeline(ir: &str) -> Result<String, String> {
    let module = LlvmModule::parse(ir)?;
    module.run_passes(PASS_PIPELINE, None)?;
    Ok(module.print())
}
