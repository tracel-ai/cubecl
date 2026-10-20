use alloc::format;
use alloc::string::{String, ToString};
use std::path::{Path, PathBuf};

use cubecl_ir::PciAddress;

use crate::utilization::{
    DevicePower, DeviceTemperature, DeviceUtilization, UtilizationUnavailable,
};

/// The amdgpu driver's files for one card: how busy it and its memory is, and its hardware
/// monitor's temperature and power sensors.
///
/// Found through the card's PCI address, never by walking `card0`, `card1`, …: a DRM card's index
/// says nothing about which device the runtime drives.
///
/// A card the kernel has suspended while idle answers every one of these with `EBUSY`, which
/// reads as the query failing until something wakes it.
#[derive(Debug)]
pub struct AmdgpuSysfsFiles {
    device: PathBuf,
    /// The card's `hwmon` directory, `None` where the driver registered none.
    hwmon: Option<PathBuf>,
}

impl AmdgpuSysfsFiles {
    pub fn new(pci_address: PciAddress) -> Self {
        let device = PathBuf::from(format!("/sys/bus/pci/devices/{pci_address}"));
        let hwmon = std::fs::read_dir(device.join("hwmon"))
            .ok()
            .and_then(|mut entries| entries.next())
            .and_then(Result::ok)
            .map(|entry| entry.path());
        Self { device, hwmon }
    }

    /// `gpu_busy_percent`, and `mem_busy_percent` beside it where the driver keeps one.
    pub fn utilization(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        let busy = read_number(&self.device.join("gpu_busy_percent"))?;
        let utilization = DeviceUtilization::new(busy);
        Ok(match read_number(&self.device.join("mem_busy_percent")) {
            Ok(memory_busy) => utilization.with_memory_busy_percent(memory_busy),
            Err(_) => utilization,
        })
    }

    /// The die's edge temperature, in degrees Celsius.
    pub fn temperature(&self) -> Result<DeviceTemperature, UtilizationUnavailable> {
        let path = self
            .hwmon
            .as_deref()
            .and_then(edge_temperature)
            .ok_or(UtilizationUnavailable::NoSuchReading)?;
        read_number(&path).map(|millidegrees| DeviceTemperature::new(millidegrees / 1000.0))
    }

    /// The card's power draw, in watts.
    pub fn power(&self) -> Result<DevicePower, UtilizationUnavailable> {
        let path = self
            .hwmon
            .as_deref()
            .and_then(|hwmon| {
                ["power1_average", "power1_input"]
                    .into_iter()
                    .map(|name| hwmon.join(name))
                    .find(|path| path.exists())
            })
            .ok_or(UtilizationUnavailable::NoSuchReading)?;
        read_number(&path).map(|microwatts| DevicePower::new(microwatts / 1_000_000.0))
    }
}

fn edge_temperature(hwmon: &Path) -> Option<PathBuf> {
    let index = (1..=8)
        .find(|index| {
            std::fs::read_to_string(hwmon.join(format!("temp{index}_label")))
                .is_ok_and(|label| label.trim() == "edge")
        })
        .unwrap_or(1);
    Some(hwmon.join(format!("temp{index}_input"))).filter(|path| path.exists())
}

fn read_number(path: &Path) -> Result<f32, UtilizationUnavailable> {
    std::fs::read_to_string(path)
        .map_err(|error| error.to_string())
        .and_then(|text| parse_number(&text))
        .map_err(|message| {
            UtilizationUnavailable::QueryFailed(format!("{}: {message}", path.display()))
        })
}

fn parse_number(text: &str) -> Result<f32, String> {
    text.trim()
        .parse::<f32>()
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A card's directory, its hardware monitor inside it, written the way amdgpu writes them.
    fn card(name: &str, device_files: &[(&str, &str)], hwmon_files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cubecl-amdgpu-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("hwmon/hwmon3")).unwrap();
        for (file, contents) in device_files {
            std::fs::write(dir.join(file), contents).unwrap();
        }
        for (file, contents) in hwmon_files {
            std::fs::write(dir.join("hwmon/hwmon3").join(file), contents).unwrap();
        }
        dir
    }

    fn files(dir: &Path) -> AmdgpuSysfsFiles {
        AmdgpuSysfsFiles {
            device: dir.to_path_buf(),
            hwmon: Some(dir.join("hwmon/hwmon3")),
        }
    }

    #[test]
    fn the_memory_is_read_in_the_same_call_as_the_device_where_it_is_kept() {
        let both = card(
            "both",
            &[("gpu_busy_percent", "37\n"), ("mem_busy_percent", "82\n")],
            &[],
        );
        assert_eq!(
            files(&both).utilization(),
            Ok(DeviceUtilization::new(37.0).with_memory_busy_percent(82.0))
        );
        let device_only = card("device-only", &[("gpu_busy_percent", "37\n")], &[]);
        assert_eq!(
            files(&device_only).utilization(),
            Ok(DeviceUtilization::new(37.0))
        );
        std::fs::remove_dir_all(both).ok();
        std::fs::remove_dir_all(device_only).ok();
    }

    #[test]
    fn the_edge_sensor_is_read_in_degrees_and_the_power_in_watts() {
        let dir = card(
            "sensors",
            &[],
            &[
                ("temp1_label", "junction\n"),
                ("temp1_input", "71000\n"),
                ("temp2_label", "edge\n"),
                ("temp2_input", "54000\n"),
                ("power1_average", "87250000\n"),
            ],
        );
        assert_eq!(files(&dir).temperature(), Ok(DeviceTemperature::new(54.0)));
        assert_eq!(files(&dir).power(), Ok(DevicePower::new(87.25)));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_sensor_the_card_lacks_is_no_such_reading() {
        let no_monitor = AmdgpuSysfsFiles {
            device: PathBuf::from("/nonexistent"),
            hwmon: None,
        };
        assert_eq!(
            no_monitor.temperature(),
            Err(UtilizationUnavailable::NoSuchReading)
        );
        let no_power = card("no-power", &[], &[("temp1_input", "40000\n")]);
        assert_eq!(
            files(&no_power).temperature(),
            Ok(DeviceTemperature::new(40.0))
        );
        assert_eq!(
            files(&no_power).power(),
            Err(UtilizationUnavailable::NoSuchReading)
        );
        std::fs::remove_dir_all(no_power).ok();
    }
}
