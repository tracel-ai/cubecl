//! The half of CUDA's compilation that runs where the server does: the PTX
//! stores, and the modules loaded from what they hold.

use crate::compiler::CudaBackend;
use crate::compute::artifact::{CudaArtifact, CudaArtifactCompiler, is_ptx_text};
use crate::compute::events::{driver_error, poisons_device};
use cubecl_core::prelude::*;
use cubecl_environment::backtrace::BackTrace;
use cubecl_llvm::nvptx::ptx_version::PtxVersion;
use cubecl_server::compiler::{
    ArtifactId, ArtifactStore, CompilationError, CompilationTarget, StoreNames,
};
use cubecl_server::kernel::{BufferIOAttr, KernelParam};
use cudarc::driver::sys::CUfunc_st;
use std::ffi::{CStr, CString, c_char};
use std::sync::Arc;

/// Loads CUDA modules for one device, from the PTX store when it holds the
/// kernel and from [`CudaArtifactCompiler`] when it does not.
#[derive(Debug)]
pub(crate) struct CudaModules {
    compiler: CudaArtifactCompiler,
    /// PTX kept by kernel and by C++ source, so a kernel expanding to text
    /// already compiled skips NVRTC. The LLVM backend emits PTX directly, so
    /// there is no source to keep it by.
    ptx_store: ArtifactStore<CudaArtifact>,
}

/// A loaded CUDA function and what launching it needs.
#[derive(Debug, Clone)]
pub struct CudaCompiledKernel {
    pub cube_dim: CubeDim,
    pub shared_mem_bytes: usize,
    pub func: *mut CUfunc_st,
    /// What the kernel does with each buffer binding, by buffer position --
    /// the compiler's answer, carried here because on a cache hit nothing
    /// else of the compilation survives. `None` for entries persisted before
    /// the answer existed, which the launch path reads as every buffer both
    /// read and written.
    pub io: Option<Arc<[BufferIOAttr]>>,
    /// The parameter list of a precompiled binary; `None` for a kernel taking
    /// `CubeCL`'s calling convention.
    pub params: Option<Arc<[KernelParam]>>,
}

impl CudaModules {
    /// `backend` is which one compiles here; see [`cache_namespace`].
    pub fn new(compiler: CudaArtifactCompiler, backend: CudaBackend) -> Self {
        let fingerprint = cache_namespace(
            &format!("ptx_sm{}", compiler.arch().version),
            backend,
            compiler.options().ptx_version,
        );

        Self {
            ptx_store: ArtifactStore::with_sources(
                StoreNames {
                    kernels: "cuda",
                    sources: "cuda-second-line",
                },
                fingerprint,
            ),
            compiler,
        }
    }
}

impl CompilationTarget for CudaModules {
    type Compiler = CudaArtifactCompiler;
    type Loaded = CudaCompiledKernel;

    fn compiler(&self) -> &CudaArtifactCompiler {
        &self.compiler
    }

    fn persists(&self) -> bool {
        self.ptx_store.persists()
    }

    fn stored(&mut self, id: &ArtifactId<()>) -> Option<CudaArtifact> {
        self.ptx_store.take(&id.kernel)
    }

    fn stored_for_source(&mut self, source: &str) -> Option<CudaArtifact> {
        self.ptx_store.take_by_source(source)
    }

    fn load(
        &mut self,
        id: &ArtifactId<()>,
        artifact: &CudaArtifact,
    ) -> Result<CudaCompiledKernel, CompilationError> {
        if is_ptx_text(&artifact.ptx) {
            dump_ptx(&id.kernel, &artifact.ptx);
        }

        let func_name = CString::new(artifact.entrypoint_name.clone()).map_err(|err| {
            CompilationError::Generic {
                reason: format!("The entrypoint name is not a C string: {err}"),
                backtrace: BackTrace::capture(),
            }
        })?;
        // SAFETY: `ptx` is a valid null-terminated PTX binary from NVRTC, or a
        // precompiled module image the kernel vouched for. `func_name` is a
        // null-terminated `CString` matching the kernel entry point in the compiled module.
        let func = unsafe {
            let module =
                cudarc::driver::result::module::load_data(artifact.ptx.as_ptr() as *const _)
                    .map_err(|err| match poisons_device(err.0) {
                        true => driver_error("cuModuleLoadData", err).into(),
                        false => CompilationError::Generic {
                            reason: format!("Unable to load the PTX: {err}"),
                            backtrace: BackTrace::capture(),
                        },
                    })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        Ok(CudaCompiledKernel {
            cube_dim: id.kernel.cube_dim.into(),
            shared_mem_bytes: artifact.shared_mem_bytes,
            func,
            io: artifact.io.clone().map(Arc::from),
            params: artifact.params.clone().map(Arc::from),
        })
    }

    /// A precompiled module is never kept: there is no compilation to save,
    /// and the store is keyed by an id only the caller vouches for.
    fn store(&mut self, id: &ArtifactId<()>, artifact: CudaArtifact, source: Option<&str>) -> bool {
        if artifact.precompiled {
            return false;
        }
        self.ptx_store.keep(&id.kernel, artifact, source)
    }
}

/// The namespace a backend's compiled artifacts live under.
///
/// Both backends emit PTX, so without the backend in the key a stale artifact from one would
/// load and run happily under the other -- tests passing while measuring nothing. The LLVM
/// backend's PTX version follows the driver, so it is in the key too: after a driver downgrade,
/// PTX newer than the driver loads would otherwise be read back and refused.
fn cache_namespace(
    fingerprint: &str,
    backend: CudaBackend,
    ptx_version: Option<PtxVersion>,
) -> String {
    match (backend, ptx_version) {
        (CudaBackend::Cpp, _) => format!("{fingerprint}-cpp"),
        (CudaBackend::Llvm, None) => format!("{fingerprint}-llvm"),
        (CudaBackend::Llvm, Some(ptx_version)) => format!("{fingerprint}-llvm-{ptx_version}"),
    }
}

/// Writes the PTX for `kernel_id` under the directory named by `CUBECL_CUDA_DUMP_PTX`, if that
/// variable is set.
///
/// Both backends end up here, so the two can be diffed instruction for instruction. That is the
/// comparison worth making: the PTX is deterministic, where a wall-clock measurement on a laptop
/// GPU is not.
fn dump_ptx(kernel_id: &KernelId, ptx: &[c_char]) {
    let Some(dir) = std::env::var_os("CUBECL_CUDA_DUMP_PTX") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    // The id is a type path plus its settings, so it carries every character a path cannot.
    let name: String = kernel_id
        .to_string()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    // ...and is long enough to blow past NAME_MAX on its own.
    let name = &name[name.len().saturating_sub(180)..];

    // SAFETY: the PTX handed to the driver is a null-terminated C string.
    let text = unsafe { CStr::from_ptr(ptx.as_ptr()) };
    let _ = std::fs::write(dir.join(format!("{name}.ptx")), text.to_bytes());
}

#[cfg(test)]
mod tests {
    use super::cache_namespace;
    use crate::compiler::CudaBackend;
    use cubecl_llvm::nvptx::ptx_version::PtxVersion;

    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    fn cache_namespace_separates_backends() {
        assert_ne!(
            cache_namespace("ptx_sm86", CudaBackend::Cpp, None),
            cache_namespace("ptx_sm86", CudaBackend::Llvm, None),
        );
    }

    /// See [`super::cache_namespace`] for why this must hold.
    #[test]
    fn cache_namespace_separates_the_llvm_backends_ptx_versions() {
        assert_ne!(
            cache_namespace("ptx_sm86", CudaBackend::Llvm, PtxVersion::for_driver(12080)),
            cache_namespace("ptx_sm86", CudaBackend::Llvm, PtxVersion::for_driver(12090)),
        );
    }
}
