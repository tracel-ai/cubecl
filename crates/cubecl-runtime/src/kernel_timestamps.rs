use cubecl_common::benchmark::ProfileDuration;
use hashbrown::HashMap;
use std::time::Instant;

use crate::server::ProfilingToken;

#[derive(Debug, Default)]
/// A simple struct to keep track of timestamps for kernel execution.
/// This should be used for servers that do not have native device profiling.
pub struct KernelTimestamps {
    start: HashMap<ProfilingToken, Instant>,
    counter: u64,
}

impl KernelTimestamps {
    /// Start measuring
    pub fn start(&mut self) -> ProfilingToken {
        let token = ProfilingToken { id: self.counter };
        self.counter += 1;
        self.start.insert(token, std::time::Instant::now());
        token
    }

    /// Stop measuring
    pub fn stop(&mut self, token: ProfilingToken) -> ProfileDuration {
        let instant = self.start.remove(&token);
        ProfileDuration::from_duration(
            instant
                .expect("Stopped timestamp before starting one.")
                .elapsed(),
        )
    }
}
