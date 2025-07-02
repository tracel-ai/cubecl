use alloc::boxed::Box;
use core::fmt::Display;

#[cfg(not(target_os = "none"))]
pub use web_time::{Duration, Instant};

#[cfg(target_os = "none")]
pub use embassy_time::{Duration, Instant};

use crate::future::DynFut;

/// How a benchmark's execution times are measured.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TimingMethod {
    /// Time measurements come from full timing of execution + sync
    /// calls.
    System,
    /// Time measurements come from hardware reported timestamps
    /// coming from a sync call.
    Device,
}

impl Display for TimingMethod {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TimingMethod::System => f.write_str("system"),
            TimingMethod::Device => f.write_str("device"),
        }
    }
}

/// Start and end point for a profile. Can be turned into a duration.
#[derive(Debug)]
pub struct ProfileTicks {
    start: Instant,
    end: Instant,
}

impl ProfileTicks {
    /// Create a new `ProfileTicks` from a start and end time.
    pub fn from_start_end(start: Instant, end: Instant) -> Self {
        Self { start, end }
    }

    /// Get the duration contained in this `ProfileTicks`.
    pub fn duration(&self) -> Duration {
        self.end.duration_since(self.start)
    }

    /// Get the duration since the epoch start of this `ProfileTicks`.
    pub fn start_duration_since(&self, epoch: Instant) -> Duration {
        self.start.duration_since(epoch)
    }

    /// Get the duration since the epoch end of this `ProfileTicks`.
    pub fn end_duration_since(&self, epoch: Instant) -> Duration {
        self.end.duration_since(epoch)
    }
}

/// Result from profiling between two measurements. This can either be a duration or a future that resolves to a duration.
pub struct ProfileDuration {
    // The future to read profiling data. For System profiling,
    // this should be entirely synchronous.
    future: DynFut<ProfileTicks>,
    method: TimingMethod,
}

impl ProfileDuration {
    /// The method used to measure the execution time.
    pub fn timing_method(&self) -> TimingMethod {
        self.method
    }

    /// Create a new `ProfileDuration` from a future that resolves to a duration.
    pub fn new(future: DynFut<ProfileTicks>, method: TimingMethod) -> ProfileDuration {
        Self { future, method }
    }

    /// Create a new `ProfileDuration` straight from a duration.
    pub fn new_system_time(start: Instant, end: Instant) -> Self {
        Self::new(
            Box::pin(async move { ProfileTicks::from_start_end(start, end) }),
            TimingMethod::System,
        )
    }

    /// Create a new `ProfileDuration` from a future that resolves to a duration.
    pub fn new_device_time(
        future: impl Future<Output = ProfileTicks> + Send + 'static,
    ) -> ProfileDuration {
        Self::new(Box::pin(future), TimingMethod::Device)
    }

    /// Retrieve the future that resolves the profile.
    pub fn into_future(self) -> DynFut<ProfileTicks> {
        self.future
    }

    /// Resolve the actual duration of the profile, possibly by waiting for the future to complete.
    pub async fn resolve(self) -> ProfileTicks {
        self.future.await
    }
}
