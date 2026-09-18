use cubecl_common::profile::{Instant, ProfileDuration};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;

use crate::server::{ProfileError, ProfilingToken, ServerError};

#[derive(Default, Debug)]
/// A simple struct to keep track of timestamps for kernel execution.
/// This should be used for servers that do not have native device profiling.
pub struct TimestampProfiler {
    state: HashMap<ProfilingToken, State>,
    counter: u64,
}

#[derive(Debug)]
enum State {
    Start(Instant),
    Error(ProfileError),
}

impl TimestampProfiler {
    /// If there is some profiling registered.
    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }
    /// Start measuring
    pub fn start(&mut self) -> ProfilingToken {
        let token = ProfilingToken { id: self.counter };
        self.counter += 1;
        self.state.insert(token, State::Start(Instant::now()));
        token
    }

    /// Stop measuring
    pub fn stop(&mut self, token: ProfilingToken) -> Result<ProfileDuration, ProfileError> {
        let state = self.state.remove(&token);
        let start = match state {
            Some(val) => match val {
                State::Start(instant) => instant,
                State::Error(profile_error) => return Err(profile_error),
            },
            None => {
                return Err(ProfileError::NotRegistered {
                    backtrace: BackTrace::capture(),
                });
            }
        };
        Ok(ProfileDuration::new_system_time(start, Instant::now()))
    }

    /// Drop the window `token` opened without measuring it, for a caller that
    /// has no way to record its end.
    ///
    /// Nothing to wait for: a system-timed window is two host instants, and
    /// the one that would be taken here is not going to be read.
    pub fn abandon(&mut self, token: ProfilingToken) {
        self.state.remove(&token);
    }

    /// Register an error during profiling.
    pub fn error(&mut self, error: ProfileError) {
        self.state
            .iter_mut()
            .for_each(|(_, state)| *state = State::Error(error.clone()));
    }

    /// Mark every open profile invalid because device work failed.
    ///
    /// This is what keeps a tuning candidate that failed from benchmarking at
    /// close to zero and winning the tune. A no-op with no profile open, so a
    /// failure path calls it unconditionally and pays nothing for the common
    /// case of no measurement in flight.
    pub fn failure(&mut self, error: &ServerError) {
        if self.state.is_empty() {
            return;
        }
        self.error(error.into());
    }
}
