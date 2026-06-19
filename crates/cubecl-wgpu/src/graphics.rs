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

impl GraphicsApi for AutoGraphicsApi {
    fn backend() -> Backend {
        // Allow overriding AutoGraphicsApi backend with ENV var in std test environments
        #[cfg(feature = "std")]
        #[cfg(test)]
        if let Ok(backend_str) = std::env::var("AUTO_GRAPHICS_BACKEND") {
            match backend_str.to_lowercase().as_str() {
                "metal" => return Backend::Metal,
                "vulkan" => return Backend::Vulkan,
                "dx12" => return Backend::Dx12,
                "opengl" => return Backend::Gl,
                "webgpu" => return Backend::BrowserWebGpu,
                _ => {
                    eprintln!(
                        "Invalid graphics backend specified in AUTO_GRAPHICS_BACKEND environment \
                         variable"
                    );
                    std::process::exit(1);
                }
            }
        }

        // In a no_std environment or if the environment variable is not set
        cfg_if::cfg_if! {
            if #[cfg(target_family = "wasm")] {
                Backend::BrowserWebGpu
            } else if #[cfg(target_os = "macos")] {
                 Backend::Metal
            } else {
                Backend::Vulkan
            }
        }
    }
}
