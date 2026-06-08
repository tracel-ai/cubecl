//! Minimal dynamic binding to NVML (the NVIDIA Management Library) used to read GPU utilization.
//!
//! `cudarc` doesn't expose NVML, and unlike memory usage, utilization is not something the runtime
//! can track itself, so we load `libnvidia-ml` at runtime and call the handful of stable,
//! decades-old C functions we need. Everything is best-effort: if the library or a symbol is
//! missing, or a call fails, we report no utilization rather than failing the server.

use core::ffi::{c_int, c_uint, c_void};
use libloading::{Library, Symbol};

/// Opaque NVML device handle (`nvmlDevice_t`).
type NvmlDevice = *mut c_void;

/// Mirror of NVML's `nvmlUtilization_t`.
#[repr(C)]
#[derive(Clone, Copy)]
struct NvmlUtilization {
    /// Percent of time over the sample period during which one or more kernels was executing.
    gpu: c_uint,
    /// Percent of time over the sample period during which device memory was read or written.
    memory: c_uint,
}

/// `NVML_SUCCESS` from `nvmlReturn_t`.
const NVML_SUCCESS: c_int = 0;

type FnInit = unsafe extern "C" fn() -> c_int;
type FnShutdown = unsafe extern "C" fn() -> c_int;
type FnHandleByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> c_int;
type FnUtilization = unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> c_int;

/// A loaded and initialized NVML instance. Calls `nvmlShutdown` on drop.
#[derive(Debug)]
pub struct Nvml {
    // Keep the library loaded for as long as we hold function pointers into it.
    _lib: Library,
    shutdown: FnShutdown,
    handle_by_index: FnHandleByIndex,
    utilization: FnUtilization,
}

impl Nvml {
    /// Load and initialize NVML, returning `None` when it isn't available at runtime.
    pub fn open() -> Option<Self> {
        // The versioned `.so.1` is what ships alongside the NVIDIA driver; the others are
        // fallbacks for less common setups and Windows.
        const NAMES: &[&str] = &["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"];

        // SAFETY: we load a system library and read well-known symbols. The function pointer
        // types below match NVML's documented, ABI-stable C signatures.
        unsafe {
            let lib = NAMES.iter().find_map(|name| Library::new(name).ok())?;

            let init: Symbol<FnInit> = lib.get(b"nvmlInit_v2\0").ok()?;
            let shutdown: Symbol<FnShutdown> = lib.get(b"nvmlShutdown\0").ok()?;
            let handle_by_index: Symbol<FnHandleByIndex> =
                lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
            let utilization: Symbol<FnUtilization> =
                lib.get(b"nvmlDeviceGetUtilizationRates\0").ok()?;

            if init() != NVML_SUCCESS {
                return None;
            }

            // Copy the raw function pointers out of the `Symbol`s (they are `Copy`) so they no
            // longer borrow `lib`, then keep `lib` alive alongside them in the returned struct.
            Some(Nvml {
                shutdown: *shutdown,
                handle_by_index: *handle_by_index,
                utilization: *utilization,
                _lib: lib,
            })
        }
    }

    /// GPU compute utilization in `0.0..=100.0` for the device at `index`, if it can be read.
    ///
    /// `index` is the NVML device index, which is assumed to match the CUDA device ordinal. This
    /// holds for the common single-GPU case and when `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
    pub fn compute_utilization(&self, index: u32) -> Option<f32> {
        // SAFETY: the function pointers were obtained from NVML in `open`, and we only read the
        // out-params after confirming each call returned `NVML_SUCCESS`.
        unsafe {
            let mut device: NvmlDevice = core::ptr::null_mut();
            if (self.handle_by_index)(index as c_uint, &mut device) != NVML_SUCCESS {
                return None;
            }

            let mut util = NvmlUtilization { gpu: 0, memory: 0 };
            if (self.utilization)(device, &mut util) != NVML_SUCCESS {
                return None;
            }

            Some(util.gpu as f32)
        }
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        // SAFETY: `shutdown` is the `nvmlShutdown` entry point obtained in `open`, balancing the
        // `nvmlInit_v2` call made there.
        unsafe {
            (self.shutdown)();
        }
    }
}
