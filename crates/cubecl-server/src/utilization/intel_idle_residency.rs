use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use cubecl_ir::PciAddress;

use crate::utilization::{DeviceUtilization, UtilizationUnavailable};

/// The milliseconds each of the card's GTs has spent in its idle power state: RC6 under i915,
/// gtidle under xe. Busy is the share of an interval the busiest GT spent out of it, which runs a
/// little above the work it ran, since a GT stays awake for a while after each burst.
///
/// The engines' own busy time is a perf event, closed to an unprivileged process.
#[derive(Debug)]
pub struct IntelIdleResidencyFiles {
    idle_residency_file_per_gt: Vec<PathBuf>,
    previous_reading: Mutex<Option<IdleResidencyReading>>,
}

#[derive(Debug)]
struct IdleResidencyReading {
    taken_at: Instant,
    idle_milliseconds_per_gt: Vec<u64>,
}

impl IntelIdleResidencyFiles {
    pub fn new(pci_address: PciAddress) -> Result<Self, UtilizationUnavailable> {
        let device_directory = PathBuf::from(format!("/sys/bus/pci/devices/{pci_address}"));
        let mut idle_residency_file_per_gt = Self::i915_rc6_residency_files(&device_directory);
        idle_residency_file_per_gt.extend(Self::xe_idle_residency_files(&device_directory));
        if idle_residency_file_per_gt.is_empty() {
            return Err(UtilizationUnavailable::QueryFailed(format!(
                "{}: holds neither i915's rc6_residency_ms nor xe's idle_residency_ms",
                device_directory.display()
            )));
        }
        Ok(Self {
            idle_residency_file_per_gt,
            previous_reading: Mutex::new(None),
        })
    }

    pub fn read(&self) -> Result<DeviceUtilization, UtilizationUnavailable> {
        let reading = self
            .read_idle_residency()
            .map_err(UtilizationUnavailable::QueryFailed)?;
        let mut previous_reading = self
            .previous_reading
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let busy_percent = previous_reading
            .as_ref()
            .and_then(|earlier| reading.busy_percent_of_busiest_gt_since(earlier));
        *previous_reading = Some(reading);
        busy_percent
            .map(DeviceUtilization::new)
            .ok_or(UtilizationUnavailable::NoPreviousReading)
    }

    fn read_idle_residency(&self) -> Result<IdleResidencyReading, String> {
        let idle_milliseconds_per_gt = self
            .idle_residency_file_per_gt
            .iter()
            .map(|path| {
                std::fs::read_to_string(path)
                    .map_err(|error| error.to_string())
                    .and_then(|text| {
                        text.trim()
                            .parse::<u64>()
                            .map_err(|error| error.to_string())
                    })
                    .map_err(|message| format!("{}: {message}", path.display()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(IdleResidencyReading {
            taken_at: Instant::now(),
            idle_milliseconds_per_gt,
        })
    }

    fn i915_rc6_residency_files(device_directory: &Path) -> Vec<PathBuf> {
        let cards = Self::numbered_entries(&device_directory.join("drm"), "card");
        let per_gt: Vec<PathBuf> = cards
            .iter()
            .flat_map(|card| Self::numbered_entries(&card.join("gt"), "gt"))
            .map(|gt| gt.join("rc6_residency_ms"))
            .filter(|path| path.exists())
            .collect();
        if !per_gt.is_empty() {
            return per_gt;
        }
        // Kernels from before the per-GT sysfs group keep one file for the whole card.
        cards
            .iter()
            .map(|card| card.join("power/rc6_residency_ms"))
            .filter(|path| path.exists())
            .collect()
    }

    fn xe_idle_residency_files(device_directory: &Path) -> Vec<PathBuf> {
        Self::numbered_entries(device_directory, "tile")
            .iter()
            .flat_map(|tile| Self::numbered_entries(tile, "gt"))
            .map(|gt| gt.join("gtidle/idle_residency_ms"))
            .filter(|path| path.exists())
            .collect()
    }

    /// `card0`, `gt1`, … : the entries of `directory` named `prefix` and a number.
    fn numbered_entries(directory: &Path, prefix: &str) -> Vec<PathBuf> {
        let Ok(entries) = std::fs::read_dir(directory) else {
            return Vec::new();
        };
        entries
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.strip_prefix(prefix))
                    .is_some_and(|number| {
                        !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit())
                    })
            })
            .map(|entry| entry.path())
            .collect()
    }
}

impl IdleResidencyReading {
    /// `None` where a GT's counter went backwards, as a driver resetting it does: the interval
    /// then measures nothing, and this reading starts the next one.
    fn busy_percent_of_busiest_gt_since(&self, earlier: &Self) -> Option<u32> {
        let elapsed_milliseconds =
            self.taken_at.duration_since(earlier.taken_at).as_secs_f64() * 1000.0;
        let mut busiest_share: f64 = 0.0;
        for (now, then) in self
            .idle_milliseconds_per_gt
            .iter()
            .zip(&earlier.idle_milliseconds_per_gt)
        {
            let idle_milliseconds = now.checked_sub(*then)?;
            busiest_share =
                busiest_share.max(1.0 - idle_milliseconds as f64 / elapsed_milliseconds);
        }
        Some((busiest_share.clamp(0.0, 1.0) * 100.0).round() as u32)
    }
}

#[cfg(test)]
mod tests {
    use core::time::Duration;

    use super::*;

    const INTERVAL: Duration = Duration::from_millis(1000);

    fn busy_percent_after_one_interval(
        idle_milliseconds_before: &[u64],
        idle_milliseconds_after: &[u64],
    ) -> Option<u32> {
        let earlier = IdleResidencyReading {
            taken_at: Instant::now(),
            idle_milliseconds_per_gt: idle_milliseconds_before.to_vec(),
        };
        let later = IdleResidencyReading {
            taken_at: earlier.taken_at + INTERVAL,
            idle_milliseconds_per_gt: idle_milliseconds_after.to_vec(),
        };
        later.busy_percent_of_busiest_gt_since(&earlier)
    }

    #[test]
    fn a_gt_reads_the_share_of_the_interval_it_spent_out_of_idle() {
        assert_eq!(busy_percent_after_one_interval(&[5000], &[6000]), Some(0));
        assert_eq!(busy_percent_after_one_interval(&[5000], &[5000]), Some(100));
        assert_eq!(busy_percent_after_one_interval(&[5000], &[5500]), Some(50));
    }

    #[test]
    fn the_busiest_gt_decides() {
        let render_idle_a_fifth = [5000, 5200];
        let media_idle_throughout = [9000, 10000];

        assert_eq!(
            busy_percent_after_one_interval(
                &[render_idle_a_fifth[0], media_idle_throughout[0]],
                &[render_idle_a_fifth[1], media_idle_throughout[1]],
            ),
            Some(80)
        );
    }

    #[test]
    fn idle_time_a_little_past_the_wall_clock_reads_as_idle() {
        assert_eq!(busy_percent_after_one_interval(&[5000], &[6003]), Some(0));
    }

    #[test]
    fn the_first_reading_and_the_first_after_a_reset_measure_nothing() {
        let directory = tempfile::tempdir().unwrap();
        let idle_residency_file = directory.path().join("idle_residency_ms");
        let files = IntelIdleResidencyFiles {
            idle_residency_file_per_gt: alloc::vec![idle_residency_file.clone()],
            previous_reading: Mutex::new(None),
        };
        let read_with_idle_milliseconds = |idle_milliseconds: &str| {
            std::fs::write(&idle_residency_file, idle_milliseconds).unwrap();
            files.read()
        };

        assert_eq!(
            read_with_idle_milliseconds("5000\n"),
            Err(UtilizationUnavailable::NoPreviousReading)
        );
        assert_eq!(
            read_with_idle_milliseconds("5000\n"),
            Ok(DeviceUtilization::new(100))
        );
        assert_eq!(
            read_with_idle_milliseconds("200\n"),
            Err(UtilizationUnavailable::NoPreviousReading)
        );
        assert_eq!(
            read_with_idle_milliseconds("200\n"),
            Ok(DeviceUtilization::new(100))
        );
    }
}
