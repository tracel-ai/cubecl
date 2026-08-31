//! Device-timing tests for the HIP backend.
//!
//! [`TimingMethod::Device`] must report the time the GPU spent inside the
//! window, not the wall-clock around a drained stream — and a window must
//! contain the windows nested inside it, which is what the nested profiling the
//! runtimes lean on reads out.

use cubecl_common::profile::{Duration, ProfileDuration, ProfileTicks, TimingMethod};
use cubecl_core::{self as cubecl, prelude::*, server::Handle};

type R = crate::HipRuntime;

#[cube(launch_unchecked)]
fn busy_kernel(output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        let mut x = output[ABSOLUTE_POS];
        for _ in 0..1024u32 {
            x = x * 1.0001 + 1.0;
        }
        output[ABSOLUTE_POS] = x;
    }
}

const N: usize = 1 << 20;

fn launch(client: &ComputeClient<R>, output: &Handle) {
    unsafe {
        busy_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static((N as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(output.clone(), N),
        );
    }
}

fn resolve(profile: ProfileDuration) -> ProfileTicks {
    assert_eq!(profile.timing_method(), TimingMethod::Device);
    cubecl_environment::future::block_on(profile.resolve())
}

#[test]
fn empty_window_reports_no_device_time() {
    let client = R::client(&Default::default());

    let (_, profile) = client.profile(|| {}, "empty").unwrap();
    let duration = resolve(profile).duration();

    // Two events recorded back to back on an idle stream: the device stamps
    // them microseconds apart, so this is "nothing" rather than exactly zero.
    assert!(
        duration < Duration::from_millis(1),
        "a window with no GPU work must measure next to nothing, got {duration:?}"
    );
}

#[test]
fn kernel_window_reports_positive_device_time() {
    let client = R::client(&Default::default());
    let output = client.empty(N * core::mem::size_of::<f32>());

    let (_, profile) = client.profile(|| launch(&client, &output), "busy").unwrap();
    let duration = resolve(profile).duration();

    assert!(duration > Duration::ZERO, "real GPU work must measure > 0");
    assert!(
        duration < Duration::from_secs(5),
        "implausibly large: {duration:?}"
    );
}

/// The nesting property: an outer window measures the work the inner ones
/// measured, and none of the profiling around them.
///
/// Under system timing this is what breaks. Every inner window has to drain the
/// stream at both ends to have anything to time, and the outer window is
/// charged for each drain, so the outer measurement grows with the number of
/// inner ones rather than with the work.
///
/// The inner windows are resolved after the outer one closes, which is how the
/// profiling logger reads them: resolving one inside the outer window would
/// stall the stream and put the stall in the outer measurement — real GPU idle
/// time, correctly reported, but not what this is testing.
#[test]
fn nested_windows_are_contained_by_the_outer_one() {
    let client = R::client(&Default::default());
    let output = client.empty(N * core::mem::size_of::<f32>());

    let (inner, outer) = client
        .profile(
            || {
                (0..4)
                    .map(|_| {
                        let (_, profile) =
                            client.profile(|| launch(&client, &output), "busy").unwrap();
                        profile
                    })
                    .collect::<Vec<_>>()
            },
            "outer",
        )
        .unwrap();

    let outer = resolve(outer).duration();
    let inner: Vec<_> = inner.into_iter().map(|p| resolve(p).duration()).collect();
    let sum: Duration = inner.iter().sum();

    for duration in &inner {
        assert!(
            *duration > Duration::ZERO,
            "an inner window measured nothing"
        );
    }
    assert!(
        outer >= sum,
        "the outer window must contain the inner ones: {outer:?} < {sum:?}"
    );
    assert!(
        outer < sum * 2,
        "the outer window is measuring the profiling rather than the work: {outer:?} vs {sum:?}"
    );
}
