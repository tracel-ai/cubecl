//! Device-timing tests for the Metal backend.
//!
//! `TimingMethod::Device` must report GPU execution time, not CPU wall-clock around a
//! synchronized dispatch. An empty window is exactly zero (no command buffers); a real
//! dispatch is positive and finite.

use cubecl_common::profile::{Duration, TimingMethod};
use cubecl_core::{self as cubecl, prelude::*};

type R = crate::MetalRuntime;

#[cube(launch_unchecked)]
fn busy_kernel(output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        let mut x = output[ABSOLUTE_POS];
        for _ in 0..256u32 {
            x = x * 1.0001 + 1.0;
        }
        output[ABSOLUTE_POS] = x;
    }
}

#[test]
fn empty_window_reports_zero_device_time() {
    let device = Default::default();
    let client = R::client(&device);

    let (_, profile) = client.profile(|| {}, "empty").unwrap();
    assert_eq!(profile.timing_method(), TimingMethod::Device);

    let ticks = cubecl_common::future::block_on(profile.resolve());
    assert_eq!(
        ticks.duration(),
        Duration::ZERO,
        "a window with no GPU work must report zero device time"
    );
}

#[test]
fn kernel_window_reports_positive_device_time() {
    let device = Default::default();
    let client = R::client(&device);

    let n = 1usize << 20;
    let output = client.empty(n * core::mem::size_of::<f32>());

    let (_, profile) = client
        .profile(
            || unsafe {
                busy_kernel::launch_unchecked::<R>(
                    &client,
                    CubeCount::Static((n as u32).div_ceil(256), 1, 1),
                    CubeDim::new_1d(256),
                    BufferArg::from_raw_parts(output.clone(), n),
                );
            },
            "busy",
        )
        .unwrap();
    assert_eq!(profile.timing_method(), TimingMethod::Device);

    let ticks = cubecl_common::future::block_on(profile.resolve());
    let dur = ticks.duration();
    assert!(dur > Duration::ZERO, "real GPU work must measure > 0");
    assert!(dur < Duration::from_secs(5), "implausibly large: {dur:?}");
}
