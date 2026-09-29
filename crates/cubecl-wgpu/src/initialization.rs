use crate::{WgpuBackend, WgpuDevice};

/// A recoverable failure while acquiring or registering a wgpu runtime.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WgpuInitError {
    /// Options, environment defaults, or supplied setup handles are invalid.
    #[error("Invalid wgpu configuration: {message}")]
    InvalidConfiguration {
        /// The invalid configuration and why it was rejected.
        message: String,
    },
    /// The requested API is not compiled for this platform.
    #[error("Graphics API {api:?} is unavailable in this build")]
    UnsupportedGraphicsApi {
        /// The requested API.
        api: WgpuBackend,
    },
    /// No adapter matches the requested selector.
    #[error("No adapter available for {device:?}: {message}")]
    AdapterUnavailable {
        /// The requested selector, including its graphics API.
        device: WgpuDevice,
        /// Details from adapter selection.
        message: String,
    },
    /// The adapter refused to create a device.
    #[error("Unable to request a wgpu device: {message}")]
    RequestDevice {
        /// Details from the graphics API.
        message: String,
    },
    /// A runtime could not be registered.
    #[error("Unable to register wgpu runtime: {message}")]
    Registration {
        /// Registration failure details.
        message: String,
    },
}
