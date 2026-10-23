//! A profile window times its own work, not the work queued ahead of it.
//!
//! Metal samples a pass's begin when its encoder starts, and orders an encoder after earlier work
//! only through a buffer the two share. Autotune queues a read past the caches before each
//! sample, a launch that shares nothing with the sample by design, and measured the sample beside
//! it: three times its own duration, except where the read happened to finish first.
//!
//! Metal only: Vulkan writes a pass's begin behind every earlier command, so it cannot fail this.
#![cfg(all(target_os = "macos", not(feature = "spirv")))]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::profile::TimingMethod;
use cubecl_core::server::Handle;
use cubecl_environment::future::block_on;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::WgpuRuntime;
use std::time::Duration;

/// Words each unit of [`read_through`] reads for the one it writes.
const READS_PER_UNIT: usize = 64;
/// Units per cube of [`read_through`].
const CUBE_UNITS: u32 = 256;

/// Read the first `READS_PER_UNIT` words of each unit's stride of `buffer` and write their sum
/// past them, into the same buffer: a launch bound by the device's memory, which two of them
/// share when they run side by side, and which binds no buffer but its own.
#[cube(launch_unchecked)]
fn read_through(buffer: &mut [u32], units: u32) {
    let units = usize::cast_from(units);
    if ABSOLUTE_POS < units {
        let mut sum = 0u32;
        for i in 0..READS_PER_UNIT {
            sum += buffer[ABSOLUTE_POS + i * units];
        }
        buffer[READS_PER_UNIT * units + ABSOLUTE_POS] = sum;
    }
}

/// One buffer of `units` strides for [`read_through`] to read, with room for what it writes.
///
/// A buffer of its own rather than an input and an output: Metal orders two launches that share
/// a buffer, and the pool can place two small outputs in one.
struct Read {
    buffer: Handle,
    units: usize,
}

impl Read {
    fn new(client: &Client, units: usize) -> Self {
        Self {
            buffer: client.empty((READS_PER_UNIT + 1) * units * size_of::<u32>()),
            units,
        }
    }

    fn launch(&self, client: &Client) {
        unsafe {
            read_through::launch_unchecked(
                client,
                CubeCount::Static((self.units as u32).div_ceil(CUBE_UNITS), 1, 1),
                CubeDim::new_1d(CUBE_UNITS),
                BufferArg::from_raw_parts(self.buffer.clone(), (READS_PER_UNIT + 1) * self.units),
                self.units as u32,
            );
        }
    }

    /// The device's time for one launch, inside a profile window.
    fn profiled(&self, client: &Client) -> Duration {
        let (_, duration) = client
            .profile(|| self.launch(client), "read_through")
            .expect("the window opens and closes");
        block_on(duration.resolve())
            .expect("the window resolves a device timing")
            .duration()
    }
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort();
    samples[samples.len() / 2]
}

#[test]
fn a_window_does_not_time_the_work_queued_ahead_of_it() {
    let client = <WgpuRuntime>::client(&Default::default());
    assert_eq!(
        client.properties().timing_method,
        TimingMethod::Device,
        "the device timer is what can see the work ahead of a window; the host's drains it"
    );

    // Several times the sample, and sharing no buffer with it: what autotune queues ahead of a
    // sample to read past the caches.
    let ahead = Read::new(&client, 1 << 20);
    let sample = Read::new(&client, 1 << 17);

    // Compiled and run once outside the measurement.
    ahead.launch(&client);
    sample.profiled(&client);

    let mut alone = Vec::new();
    let mut queued = Vec::new();
    for _ in 0..7 {
        block_on(client.sync()).expect("the device runs");
        alone.push(sample.profiled(&client));

        for _ in 0..4 {
            ahead.launch(&client);
        }
        queued.push(sample.profiled(&client));
    }
    let (alone, queued) = (median(alone), median(queued));
    eprintln!("alone {alone:?}, behind queued work {queued:?}");

    assert!(
        queued.as_secs_f64() < 1.3 * alone.as_secs_f64(),
        "a window opened behind queued work timed {queued:?}, against {alone:?} on an idle device"
    );
}
