//! Best-effort device utilization for the wgpu backend.
//!
//! wgpu exposes no utilization metric of its own, so we identify the adapter through its PCI
//! vendor/device ids (from `wgpu::AdapterInfo`) and read the figure from a vendor/OS source: the
//! amdgpu DRM `sysfs` entry on Linux for AMD, and NVML for NVIDIA. Anything else (other vendors,
//! other operating systems) reports nothing.
//!
//! Matching is by device identity rather than by a positional index: for AMD we read the busy
//! figure from the `sysfs` card whose PCI ids equal the adapter's, and for NVIDIA we only accept an
//! NVML device whose name matches the adapter's. If we can't confirm the source refers to the same
//! GPU the adapter selected, we report `None` rather than a number for the wrong device.

use cubecl_runtime::server::DeviceUtilization;

const PCI_VENDOR_AMD: u32 = 0x1002;
const PCI_VENDOR_NVIDIA: u32 = 0x10DE;

/// Compute utilization for the GPU described by `info`, if a source is available for it.
pub fn device_utilization(info: &wgpu::AdapterInfo) -> Option<DeviceUtilization> {
    let compute_percentage = match info.vendor {
        PCI_VENDOR_AMD => amd_busy_percent(info.vendor, info.device)?,
        PCI_VENDOR_NVIDIA => nvml().as_ref()?.compute_utilization(&info.name)?,
        _ => return None,
    };

    Some(DeviceUtilization { compute_percentage })
}

/// Read the amdgpu busy percentage for the DRM card whose PCI vendor/device ids match.
#[cfg(target_os = "linux")]
fn amd_busy_percent(vendor: u32, device: u32) -> Option<f32> {
    use std::fs;

    // amdgpu exposes a 0-100 busy figure per card under sysfs. Several entries (e.g. `card1` and
    // `renderD128`) point at the same device; matching on the PCI ids picks the right GPU on a
    // multi-GPU system, and any entry for that GPU yields the same figure.
    for entry in fs::read_dir("/sys/class/drm").ok()? {
        let device_dir = entry.ok()?.path().join("device");

        if read_hex(&device_dir.join("vendor")) == Some(vendor)
            && read_hex(&device_dir.join("device")) == Some(device)
        {
            let raw = fs::read_to_string(device_dir.join("gpu_busy_percent")).ok()?;
            return raw.trim().parse::<f32>().ok();
        }
    }

    None
}

#[cfg(not(target_os = "linux"))]
fn amd_busy_percent(_vendor: u32, _device: u32) -> Option<f32> {
    None
}

/// Parse a `sysfs` PCI id file such as `vendor`/`device`, whose contents look like `0x1002`.
#[cfg(target_os = "linux")]
fn read_hex(path: &std::path::Path) -> Option<u32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let trimmed = raw.trim();
    let hex = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    u32::from_str_radix(hex, 16).ok()
}

mod nvml_lib {
    //! Minimal dynamic binding to NVML, used to read NVIDIA GPU utilization. This mirrors the
    //! binding in `cubecl-cuda` but selects the device by name (matching the wgpu adapter) instead
    //! of by a CUDA ordinal, since wgpu has no ordinal that lines up with NVML.

    use core::ffi::{c_char, c_int, c_uint, c_void};
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    /// Opaque NVML device handle (`nvmlDevice_t`).
    type NvmlDevice = *mut c_void;

    /// Mirror of NVML's `nvmlUtilization_t`.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NvmlUtilization {
        gpu: c_uint,
        memory: c_uint,
    }

    /// `NVML_SUCCESS` from `nvmlReturn_t`.
    const NVML_SUCCESS: c_int = 0;

    type FnInit = unsafe extern "C" fn() -> c_int;
    type FnCount = unsafe extern "C" fn(*mut c_uint) -> c_int;
    type FnHandleByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> c_int;
    type FnName = unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> c_int;
    type FnUtilization = unsafe extern "C" fn(NvmlDevice, *mut NvmlUtilization) -> c_int;

    /// A loaded and initialized NVML instance. Lives for the lifetime of the process (held in a
    /// `OnceLock`), so we deliberately do not call `nvmlShutdown`.
    pub struct Nvml {
        _lib: Library,
        count: FnCount,
        handle_by_index: FnHandleByIndex,
        name: FnName,
        utilization: FnUtilization,
    }

    /// Process-wide NVML instance, loaded on first use. `None` when NVML isn't available.
    pub fn nvml() -> &'static Option<Nvml> {
        static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
        NVML.get_or_init(Nvml::open)
    }

    impl Nvml {
        fn open() -> Option<Self> {
            const NAMES: &[&str] = &["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"];

            // SAFETY: we load a system library and read well-known symbols. The function pointer
            // types match NVML's documented, ABI-stable C signatures.
            unsafe {
                let lib = NAMES.iter().find_map(|name| Library::new(name).ok())?;

                let init: Symbol<FnInit> = lib.get(b"nvmlInit_v2\0").ok()?;
                let count: Symbol<FnCount> = lib.get(b"nvmlDeviceGetCount_v2\0").ok()?;
                let handle_by_index: Symbol<FnHandleByIndex> =
                    lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
                let name: Symbol<FnName> = lib.get(b"nvmlDeviceGetName\0").ok()?;
                let utilization: Symbol<FnUtilization> =
                    lib.get(b"nvmlDeviceGetUtilizationRates\0").ok()?;

                if init() != NVML_SUCCESS {
                    return None;
                }

                // Copy the raw function pointers out of the `Symbol`s (they are `Copy`) so they no
                // longer borrow `lib`, then keep `lib` alive alongside them.
                Some(Nvml {
                    count: *count,
                    handle_by_index: *handle_by_index,
                    name: *name,
                    utilization: *utilization,
                    _lib: lib,
                })
            }
        }

        /// GPU compute utilization in `0.0..=100.0` for the NVML device whose name matches
        /// `adapter_name`. If there is exactly one NVML device it is used even when the names are
        /// formatted differently; otherwise an unmatched device yields `None`.
        pub fn compute_utilization(&self, adapter_name: &str) -> Option<f32> {
            // SAFETY: the function pointers were obtained from NVML in `open`. We only read
            // out-params after a `NVML_SUCCESS` return, and `name`/`utilization` write into buffers
            // we own and size below.
            unsafe {
                let mut count: c_uint = 0;
                if (self.count)(&mut count) != NVML_SUCCESS {
                    return None;
                }

                let mut single = None;
                for index in 0..count {
                    let mut device: NvmlDevice = core::ptr::null_mut();
                    if (self.handle_by_index)(index, &mut device) != NVML_SUCCESS {
                        continue;
                    }
                    if count == 1 {
                        single = Some(device);
                    }

                    let mut buffer = [0 as c_char; 96];
                    if (self.name)(device, buffer.as_mut_ptr(), buffer.len() as c_uint)
                        == NVML_SUCCESS
                        && names_match(&read_cstr(&buffer), adapter_name)
                    {
                        return self.read_utilization(device);
                    }
                }

                // A single NVIDIA GPU is unambiguous even if the names are spelled differently.
                single.and_then(|device| self.read_utilization(device))
            }
        }

        /// SAFETY: `device` must be a valid handle obtained from this NVML instance.
        unsafe fn read_utilization(&self, device: NvmlDevice) -> Option<f32> {
            let mut util = NvmlUtilization { gpu: 0, memory: 0 };
            if unsafe { (self.utilization)(device, &mut util) } != NVML_SUCCESS {
                return None;
            }
            Some(util.gpu as f32)
        }
    }

    /// Read a NUL-terminated C string from an NVML-filled buffer.
    fn read_cstr(buffer: &[c_char]) -> String {
        let bytes: Vec<u8> = buffer
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Loosely compare two GPU names, tolerating differences in case and surrounding whitespace as
    /// well as one being a prefix/suffix of the other (wgpu and NVML format names slightly
    /// differently).
    fn names_match(a: &str, b: &str) -> bool {
        let a = a.trim().to_ascii_lowercase();
        let b = b.trim().to_ascii_lowercase();
        !a.is_empty() && !b.is_empty() && (a.contains(&b) || b.contains(&a))
    }
}

use nvml_lib::nvml;
