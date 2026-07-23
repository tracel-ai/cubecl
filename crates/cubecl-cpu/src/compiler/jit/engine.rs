use std::ffi::c_void;
use std::fmt::Display;
use std::sync::{Arc, Once};

use pliron::builtin::ops::ModuleOp;
use pliron::context::Context;
use pliron_llvm::llvm_sys::core::LLVMContext;
use pliron_llvm::llvm_sys::lljit::LLVMLLJIT;
use pliron_llvm::llvm_sys::target::initialize_native;
use pliron_llvm::to_llvm_ir;

use super::data::PlironData;

/// Host ABI of a JIT'd kernel: `(buffer_ptrs, cube_count_x/y/z, unit_pos_x/y/z, metadata)`.
/// The variable-count buffers are hidden behind `buffer_ptrs` (an array of data pointers),
/// while the builtins and the metadata pointer are passed directly.
type KernelFn = extern "C" fn(*mut *mut c_void, u32, u32, u32, u32, u32, u32, *mut u64);

/// A JIT-compiled kernel. Fields drop in declaration order, so the JIT is torn down before the
/// `LLVMContext` it was built in is disposed.
#[repr(C)]
struct JitKernel {
    func: KernelFn,
    _lljit: LLVMLLJIT,
    _llvm_ctx: Arc<LLVMContext>,
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
    ) -> pliron::result::Result<Self> {
        INIT_NATIVE.call_once(|| {
            initialize_native().expect("failed to initialize native target");
        });

        let llvm_ctx = Arc::new(LLVMContext::default());
        let llvm_module = to_llvm_ir::convert_module(ctx, &llvm_ctx, module)?;

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
            _lljit: lljit,
            _llvm_ctx: llvm_ctx,
        })))
    }

    pub(crate) fn run_kernel(&self, data: &mut PlironData) {
        let b = data.builtins;
        let buffer_ptrs = data.shared.buffer_ptrs.as_ptr() as *mut *mut c_void;
        let metadata = data.shared.metadata.as_ptr() as *mut u64;
        (self.0.func)(buffer_ptrs, b[0], b[1], b[2], b[3], b[4], b[5], metadata);
    }
}

impl Display for PlironEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pliron JIT engine")
    }
}
