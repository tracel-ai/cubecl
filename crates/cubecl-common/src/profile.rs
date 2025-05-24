use alloc::boxed::Box;
use core::{fmt::Display, pin::Pin, time::Duration};

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
pub struct ProfileTicks {
    /// Start time since 'epoch' (epoch being the start of any profiling)
    start_ns: u128,
    /// End time since 'epoch' (epoch being the start of any profiling)
    end_ns: u128,
}

impl ProfileTicks {
    /// Create a new `ProfileTicks` from a start and end time.
    pub fn from_start_end(start_ns: u128, end_ns: u128) -> Self {
        Self { start_ns, end_ns }
    }

    fn duration(&self) -> Duration {
        Duration::from_nanos(self.end_ns.saturating_sub(self.start_ns) as u64)
    }
}

/// Result from profiling between two measurements. This can either be a duration or a future that resolves to a duration.
pub enum ProfileDuration {
    /// Client profile contains a full duration.
    Full(ProfileTicks),
    /// Client profile measures the device duration, and requires to be resolved.
    DeviceDuration(Pin<Box<dyn Future<Output = ProfileTicks> + Send + 'static>>),
}

impl ProfileDuration {
    /// The method used to measure the execution time.
    pub fn timing_method(&self) -> TimingMethod {
        match self {
            ProfileDuration::Full(_) => TimingMethod::System,
            ProfileDuration::DeviceDuration(_) => TimingMethod::Device,
        }
    }

    /// Create a new `ProfileDuration` straight from a duration.
    pub fn from_start_end(start_ns: u128, end_ns: u128) -> Self {
        ProfileDuration::Full(ProfileTicks { start_ns, end_ns })
    }

    /// Create a new `ProfileDuration` from a future that resolves to a duration.
    pub fn from_future(
        future: impl Future<Output = ProfileTicks> + Send + 'static,
    ) -> ProfileDuration {
        ProfileDuration::DeviceDuration(Box::pin(future))
    }

    /// Resolve the actual duration of the profile, possibly by waiting for the future to complete.
    pub async fn resolve(self) -> Duration {
        match self {
            ProfileDuration::Full(ticks) => ticks.duration(),
            ProfileDuration::DeviceDuration(future) => future.await.duration(),
        }
    }
}
