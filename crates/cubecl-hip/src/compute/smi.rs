//! Minimal dynamic binding to the `ROCm` System Management Interface (`ROCm SMI`) used to read GPU
//! utilization.
//!
//! `cubecl-hip-sys` only binds the HIP runtime, not `ROCm SMI`, and unlike memory usage, utilization
//! is not something the runtime can track itself. We therefore load `librocm_smi64` at runtime and
//! call
//! the few stable C functions we need. Everything is best-effort: if the library or a symbol is
//! missing, or a call fails, we report no utilization rather than failing the server.

use core::ffi::{c_int, c_uint};
use libloading::{Library, Symbol};

/// `RSMI_STATUS_SUCCESS` from `rsmi_status_t`.
const RSMI_STATUS_SUCCESS: c_int = 0;

type FnInit = unsafe extern "C" fn(u64) -> c_int;
type FnShutdown = unsafe extern "C" fn() -> c_int;
type FnBusyPercent = unsafe extern "C" fn(c_uint, *mut c_uint) -> c_int;

/// A loaded and initialized `ROCm SMI` instance. Calls `rsmi_shut_down` on drop.
#[derive(Debug)]
pub struct RocmSmi {
    // Keep the library loaded for as long as we hold function pointers into it.
    _lib: Library,
    shutdown: FnShutdown,
    busy_percent: FnBusyPercent,
}

impl RocmSmi {
    /// Load and initialize `ROCm SMI`, returning `None` when it isn't available at runtime.
    pub fn open() -> Option<Self> {
        // The versioned `.so.1` is what ships with `ROCm`; `.so` is the unversioned dev symlink.
        const NAMES: &[&str] = &["librocm_smi64.so.1", "librocm_smi64.so"];

        // SAFETY: we load a system library and read well-known symbols. The function pointer
        // types below match `ROCm SMI`'s documented, ABI-stable C signatures.
        unsafe {
            let lib = NAMES.iter().find_map(|name| Library::new(name).ok())?;

            let init: Symbol<FnInit> = lib.get(b"rsmi_init\0").ok()?;
            let shutdown: Symbol<FnShutdown> = lib.get(b"rsmi_shut_down\0").ok()?;
            let busy_percent: Symbol<FnBusyPercent> =
                lib.get(b"rsmi_dev_busy_percent_get\0").ok()?;

            if init(0) != RSMI_STATUS_SUCCESS {
                return None;
            }

            // Copy the raw function pointers out of the `Symbol`s (they are `Copy`) so they no
            // longer borrow `lib`, then keep `lib` alive alongside them in the returned struct.
            Some(RocmSmi {
                shutdown: *shutdown,
                busy_percent: *busy_percent,
                _lib: lib,
            })
        }
    }

    /// GPU busy percentage in `0.0..=100.0` for the device at `index`, if it can be read.
    ///
    /// `index` is the `ROCm SMI` device index, which is assumed to match the HIP device ordinal.
    /// This holds unless the two are reordered independently (e.g. via `ROCR_VISIBLE_DEVICES`).
    pub fn compute_utilization(&self, index: u32) -> Option<f32> {
        // SAFETY: `busy_percent` was obtained from `ROCm SMI` in `open`, and we only read the
        // out-param after confirming the call returned `RSMI_STATUS_SUCCESS`.
        unsafe {
            let mut busy: c_uint = 0;
            if (self.busy_percent)(index as c_uint, &mut busy) != RSMI_STATUS_SUCCESS {
                return None;
            }

            Some(busy as f32)
        }
    }
}

impl Drop for RocmSmi {
    fn drop(&mut self) {
        // SAFETY: `shutdown` is the `rsmi_shut_down` entry point obtained in `open`, balancing the
        // `rsmi_init` call made there.
        unsafe {
            (self.shutdown)();
        }
    }
}
