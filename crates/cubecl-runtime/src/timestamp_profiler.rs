use cubecl_common::profile::ProfileDuration;
use hashbrown::HashMap;
use web_time::Instant;

use crate::server::ProfilingToken;

#[derive(Default, Debug)]
/// A simple struct to keep track of timestamps for kernel execution.
/// This should be used for servers that do not have native device profiling.
pub struct TimestampProfiler {
    start: HashMap<ProfilingToken, Instant>,
    counter: u64,
}

impl TimestampProfiler {
    /// Start measuring
    pub fn start(&mut self) -> ProfilingToken {
        let token = ProfilingToken { id: self.counter };
        self.counter += 1;
        self.start.insert(token, Instant::now());
        token
    }

    /// Stop measuring
    pub fn stop(&mut self, token: ProfilingToken) -> ProfileDuration {
        let start = self
            .start
            .remove(&token)
            .expect("Stopped timestamp before starting one.");
        ProfileDuration::new_system_time(start, Instant::now())
    }
}
