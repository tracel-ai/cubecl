//! The LLJIT of the CPU target. cubecl owns it, not `pliron-llvm`, so that it can hook the
//! object layer: the JIT event listener gives the object code and its DWARF to gdb.

use crate::shared::llvm_module::{LlvmModule, error_message};
use llvm_sys::{
    execution_engine::LLVMCreateGDBRegistrationListener,
    orc2::{
        LLVMOrcCreateNewThreadSafeContextFromLLVMContext, LLVMOrcCreateNewThreadSafeModule,
        LLVMOrcDisposeThreadSafeContext, LLVMOrcDisposeThreadSafeModule,
        LLVMOrcExecutionSessionRef, LLVMOrcObjectLayerRef,
        ee::{
            LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc,
            LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener,
        },
        lljit::{
            LLVMOrcCreateLLJIT, LLVMOrcCreateLLJITBuilder, LLVMOrcDisposeLLJIT,
            LLVMOrcLLJITAddLLVMIRModule, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator,
            LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITLookup, LLVMOrcLLJITRef,
        },
    },
};
use std::ffi::{CString, c_char, c_void};

/// An LLJIT that owns the modules added to it.
pub(crate) struct Jit {
    jit: LLVMOrcLLJITRef,
}

impl Jit {
    /// A JIT for kernels. With `listeners`, the object code and its DWARF go to gdb. Without, the
    /// JIT has the default settings of LLVM.
    ///
    /// # Errors
    /// The message LLVM gives, when it cannot create the JIT for the host.
    pub(crate) fn new(listeners: bool) -> Result<Self, String> {
        let mut jit = std::ptr::null_mut();
        // SAFETY: a null builder asks for the default settings. `LLVMOrcCreateLLJIT` takes the
        // builder.
        unsafe {
            let builder = if listeners {
                let builder = LLVMOrcCreateLLJITBuilder();
                LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(
                    builder,
                    create_listened_layer,
                    std::ptr::null_mut(),
                );
                builder
            } else {
                std::ptr::null_mut()
            };
            error_message(LLVMOrcCreateLLJIT(&raw mut jit, builder))?;
        }
        Ok(Self { jit })
    }

    /// Adds `module` to the main library of the JIT.
    ///
    /// # Errors
    /// The message LLVM gives, when the JIT does not accept the module.
    pub(crate) fn add_module(&self, module: LlvmModule) -> Result<(), String> {
        let (ctx, module) = module.into_raw();
        // SAFETY: the thread-safe context takes the context, and the thread-safe module takes
        // the module. The JIT takes the thread-safe module only if the call succeeds.
        unsafe {
            let ts_ctx = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx);
            let ts_module = LLVMOrcCreateNewThreadSafeModule(module, ts_ctx);
            LLVMOrcDisposeThreadSafeContext(ts_ctx);
            let dylib = LLVMOrcLLJITGetMainJITDylib(self.jit);
            let result = error_message(LLVMOrcLLJITAddLLVMIRModule(self.jit, dylib, ts_module));
            if result.is_err() {
                LLVMOrcDisposeThreadSafeModule(ts_module);
            }
            result
        }
    }

    /// The address of the symbol `name`. The first lookup compiles the module.
    ///
    /// # Errors
    /// The message LLVM gives, when the symbol is not defined or does not compile.
    pub(crate) fn lookup(&self, name: &str) -> Result<u64, String> {
        let c_name = CString::new(name).map_err(|_| format!("symbol '{name}' contains a NUL"))?;
        let mut addr = 0;
        // SAFETY: the JIT is live, and `c_name` is a C string.
        error_message(unsafe { LLVMOrcLLJITLookup(self.jit, &mut addr, c_name.as_ptr()) })?;
        Ok(addr)
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        // SAFETY: the JIT is owned by `self`.
        if let Err(err) = error_message(unsafe { LLVMOrcDisposeLLJIT(self.jit) }) {
            log::warn!("Can't dispose the kernel JIT: {err}");
        }
    }
}

/// The object layer of a JIT with the gdb listener: `RuntimeDyld`, because the JIT event listeners
/// work only with it. It reserves one block of memory for each object, so the code and its
/// constants stay within the reach of 32-bit relocations.
extern "C" fn create_listened_layer(
    _ctx: *mut c_void,
    session: LLVMOrcExecutionSessionRef,
    _triple: *const c_char,
) -> LLVMOrcObjectLayerRef {
    // SAFETY: the session is live, and the layer takes no ownership of the listener, which is a
    // process-wide singleton in LLVM.
    unsafe {
        let layer =
            LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc(session, 1);
        LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(
            layer,
            LLVMCreateGDBRegistrationListener(),
        );
        layer
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The tests run one at a time: a JIT removes its objects from the GDB JIT interface when it
    /// is dropped.
    static JIT_TESTS: Mutex<()> = Mutex::new(());

    /// A JIT with listeners that has compiled the function `name`.
    fn listened_jit(name: &str) -> Jit {
        pliron_llvm::llvm_sys::target::initialize_native().unwrap();
        let jit = Jit::new(true).unwrap();
        let ir = format!("define i32 @{name}() {{\n  ret i32 1\n}}\n");
        jit.add_module(LlvmModule::new(&ir).unwrap()).unwrap();
        jit.lookup(name).unwrap();
        jit
    }

    /// The `RuntimeDyld` layer gives each object to gdb through the JIT event listener.
    #[test]
    fn the_listened_jit_registers_with_gdb() {
        let _guard = JIT_TESTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let name = "cubecl_gdb_rtdyld_probe";
        let _jit = listened_jit(name);
        assert!(registered_with_gdb(name));
    }

    /// An entry of the GDB JIT interface.
    #[repr(C)]
    struct JitCodeEntry {
        next: *const JitCodeEntry,
        prev: *const JitCodeEntry,
        symfile: *const u8,
        symfile_size: u64,
    }

    /// The list that gdb reads, as LLVM defines it in `JITLoaderGDB.cpp`.
    #[repr(C)]
    struct JitDescriptor {
        version: u32,
        action: u32,
        relevant: *const JitCodeEntry,
        first: *const JitCodeEntry,
    }

    unsafe extern "C" {
        static __jit_debug_descriptor: JitDescriptor;
    }

    /// Whether an object in the GDB JIT interface has the symbol `name`.
    fn registered_with_gdb(name: &str) -> bool {
        // SAFETY: LLVM changes the list only when a JIT adds or removes an object, and
        // `JIT_TESTS` keeps the JITs of this binary apart. Each entry points to a live object.
        unsafe {
            let mut entry = std::ptr::read_volatile(&raw const __jit_debug_descriptor.first);
            while !entry.is_null() {
                let object =
                    std::slice::from_raw_parts((*entry).symfile, (*entry).symfile_size as usize);
                if object
                    .windows(name.len())
                    .any(|window| window == name.as_bytes())
                {
                    return true;
                }
                entry = (*entry).next;
            }
        }
        false
    }
}
