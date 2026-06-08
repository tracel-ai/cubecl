//! Best-effort device utilization for the wgpu backend.
//!
//! wgpu exposes no utilization metric of its own, so we identify the adapter through its PCI
//! vendor/device ids (from `wgpu::AdapterInfo`) and read the figure from a vendor/OS source: the
//! amdgpu DRM `sysfs` entry on Linux for AMD, and NVML (via `nvml-wrapper`) for NVIDIA. Anything
//! else (other vendors, other operating systems) reports nothing.
//!
//! Matching is by device identity rather than by a positional index: for AMD we read the busy
//! figure from the `sysfs` card whose PCI ids equal the adapter's, and for NVIDIA we only accept an
//! NVML device whose name matches the adapter's. If we can't confirm the source refers to the same
//! GPU the adapter selected, we report `None` rather than a number for the wrong device.
//!
//! The measurement is deferred: [`device_utilization`] returns a future that performs the
//! (blocking) read when resolved, so it can be driven off the server's hot path.

use cubecl_core::future::DynFut;
use cubecl_runtime::server::DeviceUtilization;

const PCI_VENDOR_AMD: u32 = 0x1002;
const PCI_VENDOR_NVIDIA: u32 = 0x10DE;

/// Prepare a deferred utilization measurement for the GPU described by `info`.
pub fn device_utilization(info: wgpu::AdapterInfo) -> DynFut<Option<DeviceUtilization>> {
    Box::pin(async move {
        let compute_percentage = match info.vendor {
            PCI_VENDOR_AMD => amd_busy_percent(info.vendor, info.device)?,
            PCI_VENDOR_NVIDIA => nvidia_busy_percent(&info.name)?,
            _ => return None,
        };

        Some(DeviceUtilization { compute_percentage })
    })
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

/// GPU utilization for the NVML device whose name matches `adapter_name`. With a single NVIDIA GPU
/// the match is unambiguous, so the lone device is used even if names are formatted differently.
fn nvidia_busy_percent(adapter_name: &str) -> Option<f32> {
    let nvml = nvml()?;
    let count = nvml.device_count().ok()?;

    let mut matched = None;
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        if device
            .name()
            .is_ok_and(|name| names_match(&name, adapter_name))
        {
            matched = Some(device);
            break;
        }
    }

    // Fall back to the only device when there is exactly one and the name didn't match.
    let device = match matched {
        Some(device) => device,
        None if count == 1 => nvml.device_by_index(0).ok()?,
        None => return None,
    };

    device.utilization_rates().ok().map(|util| util.gpu as f32)
}

/// Process-wide NVML instance, loaded on first use. `None` when NVML isn't available at runtime.
fn nvml() -> Option<&'static nvml_wrapper::Nvml> {
    use std::sync::OnceLock;

    static NVML: OnceLock<Option<nvml_wrapper::Nvml>> = OnceLock::new();
    // NVML dynamically loads `libnvidia-ml`; it lives for the process lifetime.
    NVML.get_or_init(|| nvml_wrapper::Nvml::init().ok())
        .as_ref()
}

/// Loosely compare two GPU names, tolerating case/whitespace differences and one being a
/// prefix/suffix of the other (wgpu and NVML format names slightly differently).
fn names_match(a: &str, b: &str) -> bool {
    let a = a.trim().to_ascii_lowercase();
    let b = b.trim().to_ascii_lowercase();
    !a.is_empty() && !b.is_empty() && (a.contains(&b) || b.contains(&a))
}
