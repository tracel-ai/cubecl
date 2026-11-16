//! A [`Benchmark`] timed on several devices at once, each member on its own stream.
//!
//! [`Benchmark::run`] times one client. Work that spans devices, such as a collective, has one
//! duration per member and a slowest one that bounds the step, and must be queued on every member
//! before any is waited on, or a member blocks in the collective waiting for a peer that was never
//! launched.

use super::{Benchmark, BenchmarkComputations, BenchmarkDurations, TimingMethod};
use crate::client::Client;
use alloc::{format, string::String, vec::Vec};
use core::time::Duration;
use cubecl_common::profile::{Instant, ProfileDuration};
use cubecl_environment::future::block_on;

/// Launches the warmup makes however long they take, as [`Benchmark::run`] does.
const MIN_WARMUP_RUNS: usize = 5;

/// `benchmark` timed on every one of `clients`, member `i` being `clients[i]`.
///
/// The benchmark's [`execute`](Benchmark::execute) queues its work on every member and waits on
/// none of them. Its [`sync`](Benchmark::sync) is never called: the run syncs every client itself.
pub struct MultiDeviceRun<'a, B: Benchmark> {
    benchmark: &'a B,
    clients: &'a [Client],
}

impl<'a, B: Benchmark> MultiDeviceRun<'a, B> {
    /// A run of `benchmark` on `clients`.
    pub fn new(benchmark: &'a B, clients: &'a [Client]) -> Self {
        Self { benchmark, clients }
    }

    /// Warm the devices up for the benchmark's warmup budget, then time every member of each of
    /// its samples.
    pub fn run(&self) -> Result<MultiDeviceDurations, String> {
        let input = self.benchmark.prepare();
        let budget = self.benchmark.warmup_budget();
        let warmup = Instant::now();
        let mut warmups = 0;
        while warmups < MIN_WARMUP_RUNS || warmup.elapsed() < budget {
            self.sample(&input)?;
            warmups += 1;
        }
        let samples = (0..self.benchmark.num_samples())
            .map(|_| self.sample(&input))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MultiDeviceDurations::new(samples))
    }

    /// One execution: every member synced, a window opened on each, the work queued on all of
    /// them, every window closed, then every member synced again.
    fn sample(&self, input: &B::Input) -> Result<MultiDeviceSample, String> {
        let clients = self.clients;
        sync_all(clients)?;
        let started = Instant::now();
        let windows = clients
            .iter()
            .map(|client| client.profile_start())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("{error:?}"))?;
        let output = self.benchmark.execute(input.clone());
        let profiles = clients
            .iter()
            .zip(windows)
            .map(|(client, window)| client.profile_end(window))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("{error:?}"));
        sync_all(clients)?;
        let wall = started.elapsed();
        // Held to here so the work it names is not dropped before it runs.
        drop(output?);
        let profiles = profiles?;
        let timing_method = profiles
            .first()
            .map_or(TimingMethod::System, ProfileDuration::timing_method);
        let members = profiles
            .into_iter()
            .map(|profile| {
                block_on(profile.resolve())
                    .map(|ticks| ticks.duration())
                    .ok_or_else(|| String::from("a member's window carried no measurement"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MultiDeviceSample {
            members,
            wall,
            timing_method,
        })
    }
}

/// One execution's durations: each member's window on its own stream, and the wall around all.
#[derive(Debug, Clone)]
pub struct MultiDeviceSample {
    /// Member `i`'s window, from before its part was queued to after.
    pub members: Vec<Duration>,
    /// The host's wall from before the first window opened to every member synced.
    pub wall: Duration,
    /// How the members' windows were timed.
    pub timing_method: TimingMethod,
}

/// Every sample of a multi-device benchmark, per member, slowest member and wall.
#[derive(Debug, Clone)]
pub struct MultiDeviceDurations {
    /// Member `i`'s duration in each sample.
    pub members: Vec<BenchmarkDurations>,
    /// The slowest member's duration in each sample: what bounds a step all members take part in.
    pub slowest: BenchmarkDurations,
    /// The host's wall around each sample, syncs included.
    pub wall: BenchmarkDurations,
}

impl MultiDeviceDurations {
    /// The durations of `samples`, which all have one member count.
    ///
    /// # Panics
    ///
    /// When `samples` is empty or their member counts differ.
    pub fn new(samples: Vec<MultiDeviceSample>) -> Self {
        let first = samples
            .first()
            .expect("a multi-device benchmark takes samples");
        let timing_method = first.timing_method;
        let count = first.members.len();
        assert!(
            samples.iter().all(|sample| sample.members.len() == count),
            "every sample of a multi-device benchmark has the same members"
        );
        let column = |pick: &dyn Fn(&MultiDeviceSample) -> Duration| {
            BenchmarkDurations::from_durations(timing_method, samples.iter().map(pick).collect())
        };
        Self {
            members: (0..count)
                .map(|member| column(&|sample| sample.members[member]))
                .collect(),
            slowest: column(&|sample| sample.members.iter().copied().max().unwrap_or_default()),
            wall: BenchmarkDurations::from_durations(
                TimingMethod::System,
                samples.iter().map(|sample| sample.wall).collect(),
            ),
        }
    }

    /// The median of the slowest member, the figure a step's cost is read from.
    pub fn median(&self) -> Duration {
        BenchmarkComputations::new(&self.slowest).median
    }

    /// Slowest sample of the slowest member over its fastest: a lower bound on the run's noise.
    pub fn spread(&self) -> f64 {
        let computed = BenchmarkComputations::new(&self.slowest);
        computed.max.as_secs_f64() / computed.min.as_secs_f64().max(f64::MIN_POSITIVE)
    }
}

fn sync_all(clients: &[Client]) -> Result<(), String> {
    clients
        .iter()
        .try_for_each(|client| block_on(client.sync()).map_err(|error| format!("{error:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn sample(members: &[u64], wall: u64) -> MultiDeviceSample {
        MultiDeviceSample {
            members: members
                .iter()
                .map(|&us| Duration::from_micros(us))
                .collect(),
            wall: Duration::from_micros(wall),
            timing_method: TimingMethod::Device,
        }
    }

    #[test]
    fn the_slowest_member_of_each_sample_bounds_it() {
        let durations = MultiDeviceDurations::new(vec![
            sample(&[10, 30], 40),
            sample(&[50, 20], 60),
            sample(&[15, 25], 30),
        ]);

        let slowest: Vec<u64> = durations
            .slowest
            .durations
            .iter()
            .map(|d| d.as_micros() as u64)
            .collect();
        assert_eq!(slowest, [30, 50, 25]);
        assert_eq!(durations.members.len(), 2);
        assert_eq!(durations.median(), Duration::from_micros(30));
        assert_eq!(durations.spread(), 2.0);
    }
}
