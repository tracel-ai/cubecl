#![cfg(any(target_os = "linux", target_os = "windows"))]

//! The probes behind an autotune's roofline prediction measure a finite rate on this device.

use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_core::throughput::{MemoryAccess, MemorySpec, ThroughputKey, ThroughputMode};
use cubecl_hip::HipRuntime;
use cubecl_server::runtime::Runtime;

#[test]
fn every_probe_a_prediction_reads_measures_a_rate() {
    let client = HipRuntime::client(&Default::default());
    let modes = [
        ThroughputMode::Launch,
        ThroughputMode::SharedRead,
        ThroughputMode::CubeRoundTrips { units: 256 },
        ThroughputMode::CacheRead { bytes: 1 << 20 },
        ThroughputMode::Memory(MemorySpec::new(MemoryAccess::Copy, 1 << 20)),
        ThroughputMode::ComputeDirect {
            dtype: ElemType::Float(FloatKind::F32),
        },
    ];

    for mode in modes {
        let value =
            cubecl_std::throughput::measure_peak_throughput(&client, ThroughputKey { mode })
                .unwrap_or_else(|error| panic!("{mode:?} should measure, not fail with {error}"));
        let rate = value.ops_per_s();

        assert!(rate.is_finite() && rate > 0.0, "{mode:?} measured {rate}");
    }
}
