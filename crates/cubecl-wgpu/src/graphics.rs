pub use wgpu::Backend;
/// The basic trait to specify which graphics API to use as Backend.
///
/// Options are:
///   - [Vulkan](Vulkan)
///   - [Metal](Metal)
///   - [OpenGL](OpenGl)
///   - [DirectX 12](Dx12)
///   - [WebGpu](WebGpu)
pub trait GraphicsApi: Send + Sync + core::fmt::Debug + Default + Clone + 'static {
    /// The wgpu backend.
    fn backend() -> Backend;
}

/// Vulkan graphics API.
#[derive(Default, Debug, Clone)]
pub struct Vulkan;

/// Metal graphics API.
#[derive(Default, Debug, Clone)]
pub struct Metal;

/// OpenGL graphics API.
#[derive(Default, Debug, Clone)]
pub struct OpenGl;

/// DirectX 12 graphics API.
#[derive(Default, Debug, Clone)]
pub struct Dx12;

/// `WebGpu` graphics API.
#[derive(Default, Debug, Clone)]
pub struct WebGpu;

/// Automatic graphics API based on OS.
#[derive(Default, Debug, Clone)]
pub struct AutoGraphicsApi;

impl GraphicsApi for Vulkan {
    fn backend() -> Backend {
        Backend::Vulkan
    }
}

impl GraphicsApi for Metal {
    fn backend() -> Backend {
        Backend::Metal
    }
}

impl GraphicsApi for OpenGl {
    fn backend() -> Backend {
        Backend::Gl
    }
}

impl GraphicsApi for Dx12 {
    fn backend() -> Backend {
        Backend::Dx12
    }
}

impl GraphicsApi for WebGpu {
    fn backend() -> Backend {
        Backend::BrowserWebGpu
    }
}

impl AutoGraphicsApi {
    /// The graphics APIs to try on this machine, best first.
    ///
    /// Vulkan leads wherever it exists: it is the one that compiles to
    /// `SPIR-V`, and the rest are what a machine without it still offers.
    /// Backends this machine has no driver for enumerate nothing, so a list
    /// costs only the asking.
    ///
    /// This crate's own tests can narrow it to one with `AUTO_GRAPHICS_BACKEND`,
    /// which then holds for every `Auto` device, not only those set up through
    /// [`GraphicsApi::backend`].
    pub fn chain() -> alloc::vec::Vec<Backend> {
        #[cfg(all(feature = "std", test))]
        if let Ok(backend) = std::env::var("AUTO_GRAPHICS_BACKEND") {
            let backend = match backend.to_lowercase().as_str() {
                "metal" => Backend::Metal,
                "vulkan" => Backend::Vulkan,
                "dx12" => Backend::Dx12,
                "opengl" => Backend::Gl,
                "webgpu" => Backend::BrowserWebGpu,
                _ => {
                    eprintln!(
                        "Invalid graphics backend specified in AUTO_GRAPHICS_BACKEND environment \
                         variable"
                    );
                    std::process::exit(1);
                }
            };

            return alloc::vec![backend];
        }

        cfg_if::cfg_if! {
            if #[cfg(target_family = "wasm")] {
                alloc::vec![Backend::BrowserWebGpu]
            } else if #[cfg(target_os = "macos")] {
                alloc::vec![Backend::Metal]
            } else {
                alloc::vec![Backend::Vulkan, Backend::Dx12, Backend::Gl]
            }
        }
    }
}

impl GraphicsApi for AutoGraphicsApi {
    /// The first of the [chain](Self::chain) this machine has an adapter for —
    /// the API a [`WgpuBackend::Auto`](crate::WgpuBackend::Auto) device comes
    /// up on, so a setup made through this lands where a client made on first
    /// use would.
    fn backend() -> Backend {
        crate::runtime::resolve_backend(crate::WgpuBackend::Auto)
    }
}
