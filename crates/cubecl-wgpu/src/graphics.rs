use crate::WgpuBackend;
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

    /// The graphics API selection for `device`, before automatic selection is resolved.
    ///
    /// A named API returns its own backend. [`AutoGraphicsApi`] returns the device's backend,
    /// preserving [`WgpuBackend::Auto`] when the device does not pin an API.
    ///
    /// # Panics
    ///
    /// The default implementation rejects the noop backend, which has no `WgpuBackend` variant.
    fn backend_for(_device: &crate::WgpuDevice) -> WgpuBackend {
        match Self::backend() {
            Backend::Vulkan => WgpuBackend::Vulkan,
            Backend::Metal => WgpuBackend::Metal,
            Backend::Dx12 => WgpuBackend::Dx12,
            Backend::Gl => WgpuBackend::Gl,
            Backend::BrowserWebGpu => WgpuBackend::WebGpu,
            Backend::Noop => panic!("The noop backend has no WgpuBackend selection"),
        }
    }
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
    /// Vulkan leads wherever it reaches a GPU: it is the one that compiles to
    /// `SPIR-V`, and the rest are what a machine without it still offers. An
    /// API with only a software rasterizer is passed over for a later one
    /// with a GPU, and taken only when none has one. Backends this machine has
    /// no driver for enumerate nothing, so a list costs only the asking.
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

    /// The API `device` pins, or [`WgpuBackend::Auto`] when it pins none.
    fn backend_for(device: &crate::WgpuDevice) -> WgpuBackend {
        device.backend
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    #[test]
    fn named_apis_preserve_their_backend() {
        fn check<G: GraphicsApi>(expected: WgpuBackend) {
            let selected = G::backend_for(&crate::WgpuDevice::default());
            assert_eq!(selected, expected);
        }

        check::<Vulkan>(WgpuBackend::Vulkan);
        check::<Metal>(WgpuBackend::Metal);
        check::<Dx12>(WgpuBackend::Dx12);
        check::<OpenGl>(WgpuBackend::Gl);
        check::<WebGpu>(WgpuBackend::WebGpu);
    }

    #[test]
    fn automatic_api_preserves_auto_and_pinned_selections() {
        for backend in [
            WgpuBackend::Auto,
            WgpuBackend::Vulkan,
            WgpuBackend::Metal,
            WgpuBackend::Dx12,
            WgpuBackend::Gl,
            WgpuBackend::WebGpu,
        ] {
            let device = crate::WgpuDevice::default().on(backend);
            assert_eq!(AutoGraphicsApi::backend_for(&device), backend);
        }
    }
}
