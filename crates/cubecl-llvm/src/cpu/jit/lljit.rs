//! The LLJIT of the CPU target. cubecl owns it, not `pliron-llvm`, so that it can hook the
//! object layers: the object transform reads the symbol sizes for the perf map, and the JIT event
//! listener gives the object code and its DWARF to gdb.

use super::symbols::{JitSymbols, SymbolSizes};
use crate::shared::llvm_module::{LlvmModule, error_message};
use llvm_sys::{
    core::LLVMDisposeMessage,
    error::LLVMErrorRef,
    execution_engine::LLVMCreateGDBRegistrationListener,
    object::{
        LLVMCreateBinary, LLVMDisposeBinary, LLVMDisposeSymbolIterator, LLVMGetSymbolName,
        LLVMGetSymbolSize, LLVMMoveToNextSymbol, LLVMObjectFileCopySymbolIterator,
        LLVMObjectFileIsSymbolIteratorAtEnd,
    },
    orc2::{
        LLVMOrcCreateNewThreadSafeContextFromLLVMContext, LLVMOrcCreateNewThreadSafeModule,
        LLVMOrcDisposeThreadSafeContext, LLVMOrcDisposeThreadSafeModule,
        LLVMOrcExecutionSessionRef, LLVMOrcObjectLayerRef, LLVMOrcObjectTransformLayerSetTransform,
        ee::{
            LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManagerReserveAlloc,
            LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener,
        },
        lljit::{
            LLVMOrcCreateLLJIT, LLVMOrcCreateLLJITBuilder, LLVMOrcDisposeLLJIT,
            LLVMOrcLLJITAddLLVMIRModule, LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator,
            LLVMOrcLLJITGetMainJITDylib, LLVMOrcLLJITGetObjTransformLayer, LLVMOrcLLJITLookup,
            LLVMOrcLLJITRef,
        },
    },
    prelude::LLVMMemoryBufferRef,
};
use std::ffi::{CStr, CString, c_char, c_void};

/// An LLJIT that owns the modules added to it.
pub(crate) struct Jit {
    jit: LLVMOrcLLJITRef,
    /// The object transform writes here. It must live as long as the JIT.
    sizes: Option<Box<SymbolSizes>>,
}

impl Jit {
    /// A JIT for kernels. With `listeners`, the object code and its DWARF go to gdb. Without, the
    /// JIT has the default settings of LLVM. With the perf map in `symbols`, the JIT records the
    /// size of each symbol.
    ///
    /// # Errors
    /// The message LLVM gives, when it cannot create the JIT for the host.
    pub(crate) fn new(symbols: JitSymbols, listeners: bool) -> Result<Self, String> {
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

        let sizes = symbols.perf_map.then(|| {
            let sizes = Box::<SymbolSizes>::default();
            // SAFETY: the JIT is live, and `sizes` lives as long as the JIT (see `Drop`).
            unsafe {
                LLVMOrcObjectTransformLayerSetTransform(
                    LLVMOrcLLJITGetObjTransformLayer(jit),
                    record_symbol_sizes,
                    (&raw const *sizes).cast_mut().cast::<c_void>(),
                );
            }
            sizes
        });

        Ok(Self { jit, sizes })
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

    /// The size of the symbol `name` in the object code, if the perf map asked for it and the
    /// module is compiled.
    pub(crate) fn symbol_size(&self, name: &str) -> Option<u64> {
        self.sizes.as_ref()?.get(name)
    }
}

impl Drop for Jit {
    fn drop(&mut self) {
        // SAFETY: the JIT is owned by `self`. It is disposed before `sizes`.
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

/// The object transform: it records the size of each symbol and returns the object unchanged.
extern "C" fn record_symbol_sizes(
    ctx: *mut c_void,
    object: *mut LLVMMemoryBufferRef,
) -> LLVMErrorRef {
    // SAFETY: `ctx` is the `SymbolSizes` of the JIT (see `Jit::new`), and `object` holds a live
    // buffer. The binary reads the buffer and does not take it.
    unsafe {
        let sizes = &*(ctx as *const SymbolSizes);
        let mut message = std::ptr::null_mut();
        let binary = LLVMCreateBinary(*object, std::ptr::null_mut(), &raw mut message);
        if binary.is_null() {
            if !message.is_null() {
                LLVMDisposeMessage(message);
            }
            return std::ptr::null_mut();
        }
        let symbols = LLVMObjectFileCopySymbolIterator(binary);
        while LLVMObjectFileIsSymbolIteratorAtEnd(binary, symbols) == 0 {
            let size = LLVMGetSymbolSize(symbols);
            if size > 0 {
                let name = CStr::from_ptr(LLVMGetSymbolName(symbols));
                sizes.insert(name.to_string_lossy().into_owned(), size);
            }
            LLVMMoveToNextSymbol(symbols);
        }
        LLVMDisposeSymbolIterator(symbols);
        LLVMDisposeBinary(binary);
    }
    std::ptr::null_mut()
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
        let jit = Jit::new(JitSymbols::default(), true).unwrap();
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
