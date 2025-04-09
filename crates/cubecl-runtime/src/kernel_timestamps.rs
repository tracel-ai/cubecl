use cubecl_common::benchmark::ClientProfile;
use std::time::Instant;

#[derive(Debug, Default)]
/// A simple struct to keep track of timestamps for kernel execution.
/// This should be used for servers that do not have native device profiling.
pub struct KernelTimestamps {
    start: Option<Instant>,
}

impl KernelTimestamps {
    /// Start measuring
    pub fn start(&mut self) {
        if self.start.is_some() {
            panic!("Recursive kernel timestamps are not supported.");
        }
        self.start = Some(std::time::Instant::now());
    }

    /// Stop measuring
    pub fn stop(&mut self) -> ClientProfile {
        ClientProfile::from_duration(
            self.start
                .take()
                .expect("Stopped timestamp before starting one.")
                .elapsed(),
        )
    }
}
