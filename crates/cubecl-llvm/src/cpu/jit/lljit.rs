//! The LLJIT of the CPU target. cubecl owns it, not `pliron-llvm`, so that it can hook the
//! object layers: the object transform reads the symbol sizes for the perf map, and the JIT event
//! listener or the plugins give the object code and its DWARF to gdb and to `perf`.

#[cfg(feature = "jitdump")]
use super::perf_support::enable_perf_support;
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
use std::{
    ffi::{CStr, CString, c_char, c_void},
    sync::Once,
};

/// An LLJIT that owns the modules added to it.
pub(crate) struct Jit {
    jit: LLVMOrcLLJITRef,
    /// The object transform writes here. It must live as long as the JIT.
    sizes: Option<Box<SymbolSizes>>,
}

impl Jit {
    /// A JIT for kernels. With `listeners`, the object code and its DWARF go to gdb, and to the
    /// perf jitdump if `symbols` asks for it. Without, the JIT has the default settings of LLVM.
    ///
    /// # Errors
    /// The message LLVM gives, when it cannot create the JIT for the host.
    pub(crate) fn new(symbols: JitSymbols, listeners: bool) -> Result<Self, String> {
        // The perf support plugin needs `JITLink`, which is the default of LLVM. The gdb listener
        // needs RuntimeDyld.
        let jitdump = listeners && symbols.jitdump;
        if jitdump && cfg!(not(feature = "jitdump")) {
            static WARN: Once = Once::new();
            WARN.call_once(|| {
                log::warn!(
                    "No jitdump is written: cubecl is built without the `jitdump` feature. The \
                     perf map still names each kernel."
                );
            });
        }
        let jitlink = jitdump && cfg!(feature = "jitdump");

        let mut jit = std::ptr::null_mut();
        // SAFETY: a null builder asks for the default settings. `LLVMOrcCreateLLJIT` takes the
        // builder.
        unsafe {
            let builder = if listeners && !jitlink {
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
        #[cfg(feature = "jitdump")]
        if jitlink {
            // SAFETY: the JIT is live and has linked no object.
            unsafe { add_jitlink_plugins(jit) };
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

/// Adds the `JITLink` plugins of a JIT that writes the jitdump: the debugger plugin for gdb, and the
/// perf support plugin with the line table and the unwind data of each kernel. A plugin that
/// fails gives one warning, and the JIT continues without it.
///
/// # Safety
/// `jit` must be a live LLJIT that links with `JITLink` and has linked no object.
#[cfg(feature = "jitdump")]
unsafe fn add_jitlink_plugins(jit: LLVMOrcLLJITRef) {
    use llvm_sys::orc2::lljit::LLVMOrcLLJITEnableDebugSupport;

    // SAFETY: the caller's contract.
    if let Err(err) = error_message(unsafe { LLVMOrcLLJITEnableDebugSupport(jit) }) {
        static WARN: Once = Once::new();
        WARN.call_once(|| log::warn!("gdb does not see the kernels: {err}"));
    }
    // SAFETY: the caller's contract.
    if let Err(err) = unsafe { enable_perf_support(jit, true, true) } {
        static WARN: Once = Once::new();
        WARN.call_once(|| {
            log::warn!("No jitdump is written: {err}. The perf map still names each kernel.");
        });
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
    /// is dropped, and `JITDUMPDIR` is global.
    static JIT_TESTS: Mutex<()> = Mutex::new(());

    /// A JIT with `symbols` that has compiled the function `name`.
    fn jit_with(symbols: JitSymbols, name: &str) -> Jit {
        pliron_llvm::llvm_sys::target::initialize_native().unwrap();
        let jit = Jit::new(symbols, true).unwrap();
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
        let _jit = jit_with(JitSymbols::default(), name);
        assert!(registered_with_gdb(name));
    }

    /// With the jitdump asked for, the perf support plugin writes `jit-<pid>.dump` under
    /// `$JITDUMPDIR`, and the debugger plugin of `JITLink` gives each object to gdb. The shim
    /// fixes the records for perf: each code record comes after its own line and unwind records,
    /// the lines start at the code, and the `.eh_frame_hdr` finds the code where perf puts it.
    #[cfg(feature = "jitdump")]
    #[test]
    fn the_jitdump_is_written() {
        let _guard = JIT_TESTS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // The plugin opens its file when it is added, so `JITDUMPDIR` is set before it.
        let dir = std::env::temp_dir().join(format!("cubecl-jitdump-{}", std::process::id()));
        // SAFETY: the other tests of this binary do not read `JITDUMPDIR`, and `JIT_TESTS` keeps
        // the JIT tests apart.
        unsafe { std::env::set_var("JITDUMPDIR", &dir) };
        std::fs::create_dir_all(&dir).unwrap();

        let symbols = JitSymbols {
            perf_map: false,
            jitdump: true,
        };
        let names = ["cubecl_gdb_jitlink_probe", "cubecl_perf_second_probe"];
        pliron_llvm::llvm_sys::target::initialize_native().unwrap();
        let jit = Jit::new(symbols, true).unwrap();
        jit.add_module(LlvmModule::new(&probe_with_debug_info(&names)).unwrap())
            .unwrap();
        for name in names {
            jit.lookup(name).unwrap();
        }
        assert!(
            registered_with_gdb(names[0]),
            "the debugger plugin did not register"
        );

        let file = format!("jit-{}.dump", std::process::id());
        let path = walk(&dir).into_iter().find(|path| path.ends_with(&file));
        let dump = path.as_ref().map(|path| std::fs::read(path).unwrap());
        std::fs::remove_dir_all(&dir).ok();
        let dump = dump.unwrap_or_else(|| panic!("no {file} under {}", dir.display()));
        drop(jit);

        let offset = if super::super::perf_support::perf_adds_header_offset() {
            0
        } else {
            0x40
        };
        let loads = check_jitdump_records(&dump, offset);
        assert_eq!(loads, names.len(), "one code record for each function");
    }

    /// A module with the functions `names`, with a line table and unwind data.
    #[cfg(feature = "jitdump")]
    fn probe_with_debug_info(names: &[&str]) -> String {
        use std::fmt::Write;

        let mut ir = String::new();
        let mut metadata = String::new();
        for (i, name) in names.iter().enumerate() {
            let (program, location) = (10 + 2 * i, 11 + 2 * i);
            writeln!(
                ir,
                "define i32 @{name}() uwtable !dbg !{program} {{\n  ret i32 1, !dbg !{location}\n}}"
            )
            .unwrap();
            writeln!(
                metadata,
                "!{program} = distinct !DISubprogram(name: \"{name}\", scope: !1, file: !1, \
                 line: {program}, type: !4, scopeLine: {program}, spFlags: DISPFlagDefinition, \
                 unit: !0)\n!{location} = !DILocation(line: {location}, scope: !{program})"
            )
            .unwrap();
        }
        format!(
            "{ir}\n!llvm.dbg.cu = !{{!0}}\n!llvm.module.flags = !{{!2, !3}}\n\
             !0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: \"test\", \
             isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)\n\
             !1 = !DIFile(filename: \"probe.rs\", directory: \"/\")\n\
             !2 = !{{i32 2, !\"Debug Info Version\", i32 3}}\n\
             !3 = !{{i32 7, !\"Dwarf Version\", i32 5}}\n\
             !4 = !DISubroutineType(types: !{{}})\n{metadata}"
        )
    }

    /// Checks the records of a jitdump as `perf inject --jit` reads them, and returns the number
    /// of code records. perf gives the last line record and unwind record before a code record to
    /// that code record. `offset` is what the first line address of a function must be past its
    /// code.
    #[cfg(feature = "jitdump")]
    fn check_jitdump_records(dump: &[u8], offset: u64) -> usize {
        let u32_at = |at: usize| u32::from_le_bytes(dump[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(dump[at..at + 8].try_into().unwrap());
        let i32_in = |bytes: &[u8], at: usize| {
            i64::from(i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()))
        };

        let mut at = u32_at(8) as usize; // the size of the file header
        let mut lines = None;
        let mut unwinding: Option<(Vec<u8>, Vec<u8>)> = None;
        let mut loads = 0;
        while at + 16 <= dump.len() {
            let (id, size) = (u32_at(at), u32_at(at + 4) as usize);
            let body = at + 16;
            match id {
                0 => {
                    let (code, code_size) = (u64_at(body + 16), u64_at(body + 24));
                    let first = lines.take().expect("no line record before the code record");
                    assert_eq!(first, (code, code + offset), "the lines start at the code");

                    let (frame, header) = unwinding
                        .take()
                        .expect("no unwind record before the code record");
                    // perf puts the `.eh_frame` at `align8(T + code_size)` and the header after
                    // it. The distances do not depend on `T`.
                    let header_start = i64::try_from(frame.len()).unwrap();
                    let code_start = i64::try_from(code_size.next_multiple_of(8)).unwrap();
                    assert_eq!(header[0], 1, "the version of the `.eh_frame_hdr`");
                    assert_eq!(i32_in(&header, 4), -(header_start + 4), "`eh_frame_ptr`");
                    assert_eq!(i32_in(&header, 8), 1, "one FDE for the code");
                    let location = i32_in(&header, 12);
                    assert_eq!(location, -(code_start + header_start), "the FDE location");
                    let fde = usize::try_from(i32_in(&header, 16) + header_start).unwrap();
                    let pc_begin = i32_in(&frame, fde + 8);
                    assert_eq!(
                        pc_begin,
                        -(code_start + i64::try_from(fde).unwrap() + 8),
                        "the FDE `pc_begin`"
                    );
                    loads += 1;
                }
                2 => {
                    let code = u64_at(body);
                    assert!(u64_at(body + 8) > 0, "a line record without lines");
                    lines = Some((code, u64_at(body + 16)));
                }
                4 => {
                    let data_size = usize::try_from(u64_at(body)).unwrap();
                    let header_size = usize::try_from(u64_at(body + 8)).unwrap();
                    let data = &dump[body + 24..body + 24 + data_size];
                    let (frame, header) = data.split_at(data_size - header_size);
                    unwinding = Some((frame.to_vec(), header.to_vec()));
                }
                _ => {}
            }
            at += size;
        }
        loads
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

    #[cfg(feature = "jitdump")]
    fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        entries
            .flatten()
            .flat_map(|entry| {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path)
                } else {
                    vec![path]
                }
            })
            .collect()
    }
}
