use std::ffi::c_void;
use std::fmt::Display;
use std::rc::Rc;
use std::sync::{Arc, Once};

use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron_llvm::llvm_sys::core::LLVMContext;
use pliron_llvm::llvm_sys::lljit::LLVMLLJIT;
use pliron_llvm::llvm_sys::target::initialize_native;
use pliron_llvm::to_llvm_ir;

use super::data::PlironData;
use crate::compiler::shared_memory::SharedMemories;

/// Host ABI of a JIT'd kernel: `(buffer_ptrs, cube_count_x/y/z, unit_pos_x/y/z, sync_cube_state,
/// metadata)`. The variable-count pointers — the buffers and then the shared memories — are
/// hidden behind `buffer_ptrs`, while the builtins and both other pointers are passed directly.
type KernelFn = extern "C" fn(*mut *mut c_void, u32, u32, u32, u32, u32, u32, *mut u32, *mut u64);

/// What the host has to provide to launch a kernel, beyond its arguments.
#[derive(Clone, Debug, Default)]
pub struct KernelRequirements {
    /// Whether the kernel synchronizes its cube, in which case each of its units needs a thread
    /// of its own to run on.
    pub needs_parallelism: bool,
    /// The shared memory to reserve for a launch, and where its pointers go.
    pub shared_memories: SharedMemories,
}

/// A JIT-compiled kernel. Fields drop in declaration order, so the JIT is torn down before the
/// `LLVMContext` it was built in is disposed.
#[repr(C)]
struct JitKernel {
    func: KernelFn,
    requirements: KernelRequirements,
    _lljit: LLVMLLJIT,
    _llvm_ctx: Rc<LLVMContext>,
}

/// Safety: The kernel is immutable machine code plus the JIT/context that keep it alive.
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

/// A compiled kernel, cloneable across worker threads.
#[derive(Clone)]
pub struct PlironEngine(Arc<JitKernel>);

static INIT_NATIVE: Once = Once::new();

impl PlironEngine {
    /// Lower the LLVM-dialect module to LLVM IR and JIT-compile it with ORC/LLJIT.
    pub fn compile(
        ctx: &Context,
        module: ModuleOp,
        kernel_name: &str,
        requirements: KernelRequirements,
    ) -> pliron::result::Result<Self> {
        INIT_NATIVE.call_once(|| {
            initialize_native().expect("failed to initialize native target");
        });

        let llvm_ctx = Rc::new(LLVMContext::default());
        let llvm_module = to_llvm_ir::convert_module(ctx, &llvm_ctx, module)?;
        if std::env::var("CUBECL_DEBUG_LLVM_IR").is_ok() {
            println!("{llvm_module}");
        }

        let lljit = LLVMLLJIT::new_with_default_builder().expect("failed to create LLJIT");
        lljit
            .add_module(llvm_module)
            .expect("failed to add module to JIT");
        let addr = lljit
            .lookup_symbol(kernel_name)
            .unwrap_or_else(|err| panic!("kernel symbol '{kernel_name}' not found: {err}"));
        // Safety: the generated function is always of this form
        let func: KernelFn = unsafe { std::mem::transmute::<u64, KernelFn>(addr) };

        Ok(PlironEngine(Arc::new(JitKernel {
            func,
            requirements,
            _lljit: lljit,
            _llvm_ctx: llvm_ctx,
        })))
    }

    /// What the host has to provide to launch this kernel, see [`KernelRequirements`].
    pub(crate) fn requirements(&self) -> &KernelRequirements {
        &self.0.requirements
    }

    pub(crate) fn run_kernel(&self, data: &mut PlironData) {
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
