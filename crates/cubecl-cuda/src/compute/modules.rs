//! The half of CUDA's compilation that runs where the server does: the PTX
//! stores, and the modules loaded from what they hold.

use crate::compiler::CudaBackend;
use crate::compute::artifact::{CudaArtifact, CudaArtifactCompiler};
use crate::compute::events::{driver_error, poisons_device};
use cubecl_core::{ir::settings::DebugInfo, prelude::*};
use cubecl_environment::backtrace::BackTrace;
use cubecl_llvm::nvptx::ptx_version::PtxVersion;
use cubecl_server::compiler::{
    ArtifactId, ArtifactRoot, ArtifactStore, CompilationError, CompilationTarget, StoreNames,
};
use cubecl_server::kernel::BufferIOAttr;
use cudarc::driver::sys::CUfunc_st;
use std::ffi::{CStr, CString, c_char, c_void};
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
                    kernels: ArtifactRoot::Cuda,
                    sources: ArtifactRoot::CudaSources,
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
        dump_ptx(&id.kernel, &artifact.entrypoint_name, &artifact.ptx);

        let func_name = CString::new(artifact.entrypoint_name.clone()).unwrap();
        let line_info = id.kernel.debug_info != DebugInfo::None;
        // SAFETY: `cubin` is machine code a driver assembled for this device's architecture, and
        // `ptx` a valid null-terminated PTX binary. `func_name` is a null-terminated `CString`
        // matching the kernel entry point in the compiled module.
        let func = unsafe {
            // The machine code a compiling thread assembled. A stored environment may carry
            // machine code from a newer driver than this one, which refuses it: the PTX is
            // then assembled here instead.
            let module = cudarc::driver::result::module::load_data(
                artifact.cubin.as_ptr() as *const std::ffi::c_void
            )
            .map_err(|err| ("cuModuleLoadData", err))
            .or_else(|(op, err)| match poisons_device(err.0) {
                true => Err((op, err)),
                false => load_module(artifact.ptx.as_ptr().cast(), line_info)
                    .map_err(|err| (load_module_op(line_info), err)),
            })
            .map_err(|(op, err)| {
                if poisons_device(err.0) {
                    driver_error(op, err).into()
                } else {
                    CompilationError::Generic {
                        reason: format!("Unable to load the PTX: {err}"),
                        backtrace: BackTrace::capture(),
                    }
                }
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
        })
    }

    fn store(&mut self, id: &ArtifactId<()>, artifact: CudaArtifact, source: Option<&str>) -> bool {
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

/// Loads the module of `ptx`. With `line_info`, the driver keeps the `.loc` lines of the PTX in the
/// code, for Nsight Compute and CUPTI (`CU_JIT_GENERATE_LINE_INFO`). The optimization level does
/// not change.
///
/// # Safety
/// `ptx` must be a valid null-terminated PTX image.
unsafe fn load_module(
    ptx: *const c_void,
    line_info: bool,
) -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    use cudarc::driver::sys;

    if !line_info {
        // SAFETY: the caller's contract.
        return unsafe { cudarc::driver::result::module::load_data(ptx) };
    }
    let mut module = std::mem::MaybeUninit::uninit();
    let mut options = [sys::CUjit_option::CU_JIT_GENERATE_LINE_INFO];
    // The value of a flag option is the integer itself, in the place of a pointer.
    let mut values = [std::ptr::without_provenance_mut::<c_void>(1)];
    // SAFETY: the caller's contract, and the two arrays have one element each, as the count says.
    unsafe {
        sys::cuModuleLoadDataEx(
            module.as_mut_ptr(),
            ptx,
            1,
            options.as_mut_ptr(),
            values.as_mut_ptr(),
        )
        .result()?;
        Ok(module.assume_init())
    }
}

/// The driver call that [`load_module`] makes, to name it in an error.
fn load_module_op(line_info: bool) -> &'static str {
    if line_info {
        "cuModuleLoadDataEx"
    } else {
        "cuModuleLoadData"
    }
}

/// Writes the PTX for `kernel_id` under the directory named by `CUBECL_CUDA_DUMP_PTX`, if that
/// variable is set, in a file named by [`dump_file_name`].
///
/// Both backends end up here, so the two can be diffed instruction for instruction. That is the
/// comparison worth making: the PTX is deterministic, where a wall-clock measurement on a laptop
/// GPU is not.
fn dump_ptx(kernel_id: &KernelId, entrypoint: &str, ptx: &[c_char]) {
    let Some(dir) = std::env::var_os("CUBECL_CUDA_DUMP_PTX") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    // SAFETY: the PTX handed to the driver is a null-terminated C string.
    let text = unsafe { CStr::from_ptr(ptx.as_ptr()) };
    let name = dump_file_name(kernel_id, entrypoint);
    let _ = std::fs::write(dir.join(format!("{name}.ptx")), text.to_bytes());
}

/// The longest file name [`dump_file_name`] returns, the `.ptx` extension aside: well under
/// `NAME_MAX` (255 bytes on the usual Linux file systems).
const DUMP_NAME_MAX: usize = 180;

/// The name, without extension, of the file the PTX of `kernel_id` is dumped to.
///
/// It is the kernel's entry point: its name and comptime arguments first, then the discriminator
/// of [`KernelId::entrypoint_name`], a hash of the whole id. Every comptime variant of a kernel
/// therefore gets its own file, named after the function the PTX defines. The id's `Display`
/// does not do: it leaves the kernel's name out and ends with the names of the runtime
/// arguments, which every variant shares.
///
/// The discriminator is added when the entry point does not already end with it, and shortening
/// cuts the front part, so the hash always survives.
fn dump_file_name(kernel_id: &KernelId, entrypoint: &str) -> String {
    let discriminator = kernel_id.entrypoint_name("");
    let base = entrypoint
        .strip_suffix(&discriminator)
        .unwrap_or(entrypoint);
    // An entry point is a C identifier, but sanitize anyway: a path must not escape the directory.
    let mut name: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .take(DUMP_NAME_MAX - discriminator.len())
        .collect();
    name.push_str(&discriminator);
    name
}

#[cfg(test)]
mod tests {
    use super::{DUMP_NAME_MAX, cache_namespace, dump_file_name};
    use crate::compiler::CudaBackend;
    use cubecl_core::prelude::KernelId;
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

    /// Comptime variants of one kernel share the id's runtime argument names; their dumps must
    /// not overwrite each other.
    #[test]
    fn dump_file_name_separates_comptime_variants() {
        struct Kernel;
        let a = KernelId::new::<Kernel>().info((1u32, "f32"));
        let b = KernelId::new::<Kernel>().info((2u32, "f32"));

        let name_a = dump_file_name(&a, &a.entrypoint_name("factor_kernel_f_f32"));
        let name_b = dump_file_name(&b, &b.entrypoint_name("factor_kernel_f_f32"));

        assert_ne!(name_a, name_b);
        assert!(name_a.starts_with("factor_kernel_f_f32_"), "{name_a}");
        assert_eq!(name_a, a.entrypoint_name("factor_kernel_f_f32"));
    }

    /// An entry point without the id's discriminator, such as a fixed name, still gets one.
    #[test]
    fn dump_file_name_adds_a_missing_discriminator() {
        struct Kernel;
        let a = KernelId::new::<Kernel>().info(1u32);
        let b = KernelId::new::<Kernel>().info(2u32);

        assert_eq!(dump_file_name(&a, "main"), a.entrypoint_name("main"));
        assert_ne!(dump_file_name(&a, "main"), dump_file_name(&b, "main"));
    }

    /// A long entry point is cut at the end of its name, never through the discriminator, and
    /// nothing in it can name another directory.
    #[test]
    fn dump_file_name_keeps_the_discriminator_when_shortening() {
        struct Kernel;
        let id = KernelId::new::<Kernel>().info(7u32);
        let discriminator = id.entrypoint_name("");
        let long = format!("../{}", "k".repeat(400));

        let name = dump_file_name(&id, &long);

        assert_eq!(name.len(), DUMP_NAME_MAX);
        assert!(name.ends_with(&discriminator), "{name}");
        assert!(name.starts_with("___k"), "{name}");
        assert!(!name.contains('/') && !name.contains('.'), "{name}");
    }
}
