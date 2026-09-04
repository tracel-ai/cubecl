//! What a backend reporting [`TimingMethod::Device`] promises.
//!
//! Device timing must report the time the GPU spent inside the window, not the
//! wall clock around a drained stream. The nesting property is separate
//! ([`testgen_profiling_nested`](crate::testgen_profiling_nested)) because a
//! backend can honor the first without honoring it.

use crate as cubecl;
use alloc::vec::Vec;
use cubecl_runtime::runtime::Runtime;

use cubecl::prelude::*;
use cubecl_common::profile::{Duration, ProfileDuration, TimingMethod};
use cubecl_runtime::server::Handle;

/// Long enough that the window has something to measure on any device, short
/// enough that a nesting test running four of them stays quick.
const ITERATIONS: u32 = 1024;

/// One element per unit; large enough that the launch is not all latency.
const LEN: usize = 1 << 20;

#[cube(launch_unchecked)]
fn busy_kernel(output: &mut [f32]) {
    if ABSOLUTE_POS < output.len() {
        let mut x = output[ABSOLUTE_POS];
        for _ in 0..ITERATIONS {
            x = x * 1.0001 + 1.0;
        }
        output[ABSOLUTE_POS] = x;
    }
}

/// Launch counts sixteen times apart, both long enough to fill several command
/// passes on a backend that batches launches into them.
const FEW_LAUNCHES: usize = 32;
const MANY_LAUNCHES: usize = 512;

/// A quarter of the sixteen those counts differ by, leaving room for timer noise
/// and the fixed cost of a window, and still far above the ratio of one that a
/// window pinned to its first pass reports.
const MIN_GROWTH: u32 = 4;

#[cube(launch_unchecked)]
fn touch_kernel(output: &mut [f32]) {
    if ABSOLUTE_POS == 0 {
        output[0] += 1.0;
    }
}

fn touch(client: &Client, output: &Handle) {
    unsafe {
        touch_kernel::launch_unchecked(
            client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            BufferArg::from_raw_parts(output.clone(), 1),
        );
    }
}

fn launch(client: &Client, output: &Handle) {
    unsafe {
        busy_kernel::launch_unchecked(
            client,
            CubeCount::Static((LEN as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(output.clone(), LEN),
        );
    }
}

fn resolve(profile: ProfileDuration) -> Duration {
    assert_eq!(profile.timing_method(), TimingMethod::Device);
    cubecl_environment::future::block_on(profile.resolve()).duration()
}

/// A window with no GPU work in it measures next to nothing.
///
/// The bound is what separates device time from host time: a backend that
/// times the wall clock around a drained stream reports the drain here, which
/// is milliseconds of launch latency rather than the microseconds two events
/// recorded back to back on an idle stream are apart.
pub fn test_empty_window_reports_no_device_time<R: Runtime>(client: Client) {
    let (_, profile) = client.profile(|| {}, "empty").unwrap();
    let duration = resolve(profile);

    assert!(
        duration < Duration::from_millis(1),
        "a window with no GPU work must measure next to nothing, got {duration:?}"
    );
}

/// Real GPU work measures positive and plausible.
///
/// The upper bound is the one that catches a broken clock conversion: a
/// backend reading its device's ticks as the wrong unit passes "> 0" and fails
/// here.
pub fn test_kernel_window_reports_positive_device_time<R: Runtime>(client: Client) {
    let output = client.empty(LEN * core::mem::size_of::<f32>());

    let (_, profile) = client.profile(|| launch(&client, &output), "busy").unwrap();
    let duration = resolve(profile);

    assert!(duration > Duration::ZERO, "real GPU work must measure > 0");
    assert!(
        duration < Duration::from_secs(5),
        "implausibly large: {duration:?}"
    );
}

/// A window measures every command pass it spans, not just the first.
///
/// A backend batches launches into passes of a fixed size, so a loop long
/// enough to fill several is the ordinary case rather than an edge one. Marking
/// only the pass that opened the window reports the same figure however long
/// the loop runs, which reads as launches getting cheaper the more of them
/// there are, and autotune prices a candidate's launches off it.
pub fn test_window_spans_every_pass_in_it<R: Runtime>(client: Client) {
    let output = client.empty(core::mem::size_of::<f32>());

    // Compiling the kernel would otherwise land inside the first window.
    touch(&client, &output);
    cubecl_environment::future::block_on(client.sync()).unwrap();

    let window = |launches: usize| {
        let (_, profile) = client
            .profile(
                || {
                    for _ in 0..launches {
                        touch(&client, &output);
                    }
                },
                "touch",
            )
            .unwrap();
        resolve(profile)
    };

    let few = window(FEW_LAUNCHES);
    let many = window(MANY_LAUNCHES);

    assert!(
        many > few * MIN_GROWTH,
        "{MANY_LAUNCHES} launches measured {many:?} against {few:?} for {FEW_LAUNCHES}"
    );
}

/// An outer window measures the work the inner ones measured, and none of the
/// profiling around them.
///
/// This is what system timing cannot do, and the reason a backend implements
/// device timing at all. Every inner window has to drain the stream at both
/// ends to have anything to time, and the outer window is charged for each
/// drain, so the outer measurement grows with the number of inner windows
/// rather than with the work.
///
/// The inner windows resolve after the outer one closes, which is how the
/// profiling logger reads them: resolving one inside the outer window would
/// stall the stream and put the stall in the outer measurement — real GPU idle
/// time, correctly reported, but not what this is testing.
pub fn test_nested_windows_are_contained_by_the_outer_one<R: Runtime>(client: Client) {
    let output = client.empty(LEN * core::mem::size_of::<f32>());

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

    let outer = resolve(outer);
    let inner: Vec<_> = inner.into_iter().map(resolve).collect();
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

/// The device-timing contract, for every backend that reports
/// [`TimingMethod::Device`](cubecl_common::profile::TimingMethod::Device).
#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_profiling {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_empty_window_reports_no_device_time() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::profiling::test_empty_window_reports_no_device_time::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_kernel_window_reports_positive_device_time() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::profiling::test_kernel_window_reports_positive_device_time::<
                TestRuntime,
            >(client);
        }

        #[$crate::runtime_tests::test_log::test]
        fn test_window_spans_every_pass_in_it() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::profiling::test_window_spans_every_pass_in_it::<
                TestRuntime,
            >(client);
        }
    };
}

/// The nesting property, for a backend whose profiler tracks more than one
/// open window at a time.
///
/// Separate from [`testgen_profiling`] because a backend can report honest
/// device time for a single window without supporting nested ones, and the
/// difference is a property of the profiler rather than of the device.
#[allow(missing_docs)]
#[macro_export]
macro_rules! testgen_profiling_nested {
    () => {
        use super::*;

        #[$crate::runtime_tests::test_log::test]
        fn test_nested_windows_are_contained_by_the_outer_one() {
            let client = TestRuntime::client(&Default::default());
            cubecl_core::runtime_tests::profiling::test_nested_windows_are_contained_by_the_outer_one::<
                TestRuntime,
            >(client);
        }
    };
}
