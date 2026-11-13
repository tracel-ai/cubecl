//! The half of HIP's compilation that runs where the server does: the
//! compilation stores, and the modules loaded from what they hold.

use crate::compiler::HipBackend;
use crate::compute::artifact::{HipArtifact, HipArtifactCompiler};
use crate::compute::status::checked;
use cubecl_core::prelude::*;
use cubecl_server::compiler::{
    ArtifactId, ArtifactRoot, ArtifactStore, CompilationError, CompilationTarget, StoreNames,
};
use cubecl_server::kernel::BufferIOAttr;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// Names an architecture to compile every kernel for in place of the device's, without
/// launching any: how a part that is not here is checked for what its kernels compile to.
pub(crate) const COMPILE_FOR_VAR: &str = "CUBECL_HIP_COMPILE_FOR";

/// Where the code objects compiled for [`COMPILE_FOR_VAR`] are written, a directory under the
/// system's temporary one when unset.
const DUMP_DIR_VAR: &str = "CUBECL_HIP_DUMP_DIR";

/// The directory kernels compiled for another architecture go to, or `None` when they are
/// compiled for the device and launched.
fn compile_only_dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let arch = std::env::var(COMPILE_FOR_VAR).ok()?;
        let dir = std::env::var_os(DUMP_DIR_VAR)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join(format!("cubecl-hip-{arch}")));
        std::fs::create_dir_all(&dir).expect("the dump directory can be created");
        Some(dir)
    })
    .as_ref()
}

/// Writes `artifact`'s code object to `dir`, beside the kernel it was compiled for.
fn dump(dir: &std::path::Path, id: &ArtifactId<()>, artifact: &HipArtifact) {
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let index = COUNT.fetch_add(1, Ordering::Relaxed);
    let stem = format!("{index:05}-{}", artifact.entrypoint_name);
    std::fs::write(dir.join(format!("{stem}.hsaco")), &artifact.binary)
        .expect("the code object can be written");
    std::fs::write(dir.join(format!("{stem}.txt")), format!("{:#?}", id.kernel))
        .expect("the kernel description can be written");
}

/// Loads HIP modules for one device, from the compilation store when it holds
/// the kernel and from [`HipArtifactCompiler`] when it does not.
#[derive(Debug)]
pub(crate) struct HipModules {
    compiler: HipArtifactCompiler,
    /// Code objects kept by kernel and by C++ source, so a kernel expanding
    /// to text already compiled skips HIP RTC. The LLVM backend emits a code
    /// object directly, so there is no source to keep it by.
    store: ArtifactStore<HipArtifact>,
}

/// A loaded HIP module and what launching its kernel needs.
#[derive(Debug, Clone)]
pub struct HipCompiledKernel {
    /// The module `func` belongs to. Never unloaded: see
    /// [`HipContext::new`](super::context::HipContext::new).
    #[allow(
        dead_code,
        reason = "kept beside the function it owns; nothing unloads it"
    )]
    pub module: cubecl_hip_sys::hipModule_t,
    pub func: cubecl_hip_sys::hipFunction_t,
    pub cube_dim: CubeDim,
    pub shared_mem_bytes: usize,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the compiler's answer, carried here because on a cache hit nothing
    /// else of the compilation survives. `None` for entries persisted before
    /// the answer existed, which the launch path reads as every buffer both
    /// read and written.
    pub io: Option<Arc<[BufferIOAttr]>>,
}

impl HipModules {
    /// `fingerprint` is the one the runtime already published on
    /// [`DeviceProperties::identity`], rather than one rebuilt here: the
    /// namespace a kernel is cached under and the identity a bundle is stamped
    /// with have to be the same string, and the only way to guarantee that is
    /// for there to be one string.
    ///
    /// `backend` is which one compiles here; see [`cache_namespace`].
    ///
    /// [`DeviceProperties::identity`]: cubecl_core::ir::DeviceProperties::identity
    pub fn new(compiler: HipArtifactCompiler, fingerprint: String, backend: HipBackend) -> Self {
        let fingerprint = cache_namespace(&fingerprint, backend);

        Self {
            compiler,
            store: ArtifactStore::with_sources(
                StoreNames {
                    kernels: ArtifactRoot::Hip,
                    sources: ArtifactRoot::HipSources,
                },
                fingerprint,
            ),
        }
    }
}

impl CompilationTarget for HipModules {
    type Compiler = HipArtifactCompiler;
    type Loaded = HipCompiledKernel;

    fn compiler(&self) -> &HipArtifactCompiler {
        &self.compiler
    }

    fn persists(&self) -> bool {
        self.store.persists()
    }

    fn stored(&mut self, id: &ArtifactId<()>) -> Option<HipArtifact> {
        self.store.take(&id.kernel)
    }

    fn stored_for_source(&mut self, source: &str) -> Option<HipArtifact> {
        self.store.take_by_source(source)
    }

    fn load(
        &mut self,
        id: &ArtifactId<()>,
        artifact: &HipArtifact,
    ) -> Result<HipCompiledKernel, CompilationError> {
        if let Some(dir) = compile_only_dir() {
            dump(dir, id, artifact);
            return Ok(HipCompiledKernel {
                module: std::ptr::null_mut(),
                func: std::ptr::null_mut(),
                cube_dim: id.kernel.cube_dim.into(),
                shared_mem_bytes: artifact.shared_mem_bytes,
                io: artifact.io.clone().map(Arc::from),
            });
        }
        let func_name = CString::new(artifact.entrypoint_name.clone()).unwrap();

        // Create the HIP module
        let mut module: cubecl_hip_sys::hipModule_t = std::ptr::null_mut();
        // SAFETY: `code` contains a valid compiled binary obtained from `hiprtcGetCode`.
        // `module` receives the loaded module handle on success.
        unsafe {
            let codeptr = artifact.binary.as_ptr();
            let status = cubecl_hip_sys::hipModuleLoadData(&mut module, codeptr as *const _);
            checked("hipModuleLoadData", status)?;
        }
        // Retrieve the HIP module function
        let mut func: cubecl_hip_sys::hipFunction_t = std::ptr::null_mut();
        // SAFETY: `module` is a valid loaded module from `hipModuleLoadData` above.
        // `func_name` is a null-terminated `CString` matching the kernel entry point.
        unsafe {
            let status =
                cubecl_hip_sys::hipModuleGetFunction(&mut func, module, func_name.as_ptr());
            checked("hipModuleGetFunction", status)?;
        }

        Ok(HipCompiledKernel {
            module,
            func,
            cube_dim: id.kernel.cube_dim.into(),
            shared_mem_bytes: artifact.shared_mem_bytes,
            io: artifact.io.clone().map(Arc::from),
        })
    }

    fn store(&mut self, id: &ArtifactId<()>, artifact: HipArtifact, source: Option<&str>) -> bool {
        self.store.keep(&id.kernel, artifact, source)
    }
}

/// The namespace a backend's compiled artifacts live under.
///
/// Both backends emit AMD code objects, so without the backend in the key a stale
/// artifact from one would load and run happily under the other — tests passing
/// while measuring nothing.
fn cache_namespace(fingerprint: &str, backend: HipBackend) -> String {
    let backend = match backend {
        HipBackend::Cpp => "cpp",
        HipBackend::Llvm => "llvm",
    };
    format!("{fingerprint}-{backend}")
}

#[cfg(test)]
mod tests {
    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    fn cache_namespace_separates_backends() {
        assert_ne!(
            super::cache_namespace("gfx1201-abc", crate::compiler::HipBackend::Cpp),
            super::cache_namespace("gfx1201-abc", crate::compiler::HipBackend::Llvm),
        );
    }
}
