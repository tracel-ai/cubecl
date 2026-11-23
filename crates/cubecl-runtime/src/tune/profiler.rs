use crate::client::{Client, ProfileWindow};
use crate::execution::{StreamMode, StreamModeOverride};
use crate::server::{ProfileError, ServerError};
use alloc::boxed::Box;
use alloc::vec::Vec;
use cubecl_common::profile::{ProfileDuration, ProfileTicks};

/// Runs a tune's candidates and times them, on the device that runs them or on the devices that
/// run each of them together.
#[derive(Clone)]
pub(crate) struct CandidateProfiler {
    dispatch: CandidateProfilerDispatch,
}

#[derive(Clone)]
enum CandidateProfilerDispatch {
    Device(DeviceProfiler),
    Group(GroupProfiler),
}

impl CandidateProfiler {
    /// Times candidates on the devices of `clients`, the first of which answers for the tune.
    ///
    /// # Panics
    ///
    /// Without a client.
    pub(crate) fn new(clients: &[Client]) -> Self {
        assert!(!clients.is_empty(), "a tune runs on a device at least");
        let dispatch = if clients.len() == 1 {
            CandidateProfilerDispatch::Device(DeviceProfiler::new(&clients[0]))
        } else {
            CandidateProfilerDispatch::Group(GroupProfiler::new(clients))
        };
        Self { dispatch }
    }

    /// The client the tune is marked compiled or lost under.
    pub(crate) fn lead(&self) -> &Client {
        &self.clients()[0]
    }

    /// Runs `task` with nothing else landing on the device, where one device holds the candidates.
    ///
    /// # Errors
    ///
    /// The panic of `task`, where a device thread runs it.
    pub(crate) fn exclusive<R: Send + 'static>(
        &self,
        task: impl FnOnce() -> R + Send,
    ) -> Result<R, ServerError> {
        match &self.dispatch {
            CandidateProfilerDispatch::Device(device) => device.exclusive(task),
            CandidateProfilerDispatch::Group(group) => group.exclusive(task),
        }
    }

    /// Runs `candidate` and times it, the slowest device's time where several run it.
    ///
    /// # Errors
    ///
    /// Why a device opened or closed no window.
    pub(crate) fn profile<O: Send + 'static>(
        &self,
        candidate: impl FnOnce() -> O + Send,
        name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        match &self.dispatch {
            CandidateProfilerDispatch::Device(device) => device.profile(candidate, name),
            CandidateProfilerDispatch::Group(group) => group.profile(candidate),
        }
    }

    /// Puts this thread's stream of every device in `mode` until the guards drop: a candidate of
    /// several devices that ran on some of them alone would wait for the others.
    pub(crate) fn stream_modes(&self, mode: StreamMode) -> Vec<StreamModeOverride> {
        self.clients()
            .iter()
            .map(|client| StreamModeOverride::new(mode, client))
            .collect()
    }

    /// Compiles the kernels queued on every device.
    pub(crate) fn compile_queued(&self) {
        self.clients().iter().for_each(Client::compile_queued);
    }

    /// Flushes every device, dropping what earlier work failed with.
    pub(crate) fn flush(&self) {
        for client in self.clients() {
            let _errs = client.flush();
        }
    }

    /// Waits for every device, and hands back the fault of the first one that is lost.
    ///
    /// # Errors
    ///
    /// The fault.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn expect_devices(&self) -> Result<(), ServerError> {
        self.clients()
            .iter()
            .map(|client| cubecl_environment::future::block_on(client.sync()))
            .find(|synced| matches!(synced, Err(err) if err.is_device_poisoned()))
            .unwrap_or(Ok(()))
    }

    fn clients(&self) -> &[Client] {
        match &self.dispatch {
            CandidateProfilerDispatch::Device(device) => device.clients(),
            CandidateProfilerDispatch::Group(group) => group.clients(),
        }
    }
}

/// Times candidates one device runs alone, the device held for each.
#[derive(Clone)]
struct DeviceProfiler {
    client: Client,
}

impl DeviceProfiler {
    pub fn new(client: &Client) -> Self {
        Self {
            client: client.clone(),
        }
    }

    pub fn clients(&self) -> &[Client] {
        core::slice::from_ref(&self.client)
    }

    pub fn exclusive<R: Send + 'static>(
        &self,
        task: impl FnOnce() -> R + Send,
    ) -> Result<R, ServerError> {
        self.client.exclusive(task)
    }

    pub fn profile<O: Send + 'static>(
        &self,
        candidate: impl FnOnce() -> O + Send,
        name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        self.client.profile(candidate, name)
    }
}

/// Times candidates several devices run together.
///
/// No device is held: holding one runs the candidate on that device's thread, where a candidate
/// that waits for another device's thread can wait for itself. A window is opened on each device
/// instead, at the first kernel the candidate launches there, so the host reaching the devices one
/// after another is left out of every window but the wait it causes on the devices reached first.
#[derive(Clone)]
struct GroupProfiler {
    clients: Vec<Client>,
}

impl GroupProfiler {
    pub fn new(clients: &[Client]) -> Self {
        Self {
            clients: clients.to_vec(),
        }
    }

    pub fn clients(&self) -> &[Client] {
        &self.clients
    }

    pub fn exclusive<R>(&self, task: impl FnOnce() -> R) -> Result<R, ServerError> {
        Ok(task())
    }

    /// Every device's queue is flushed before any window closes: closing one may wait for its
    /// device, which waits for what the others were asked to launch.
    pub fn profile<O>(
        &self,
        candidate: impl FnOnce() -> O,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        let windows = OpenWindows::open(&self.clients)?;
        let out = candidate();
        for client in &self.clients {
            let _errs = client.flush();
        }
        let durations = windows.close()?;
        Ok((out, slowest(durations)))
    }
}

/// The window open on each device, abandoned where it was not closed: a candidate that unwinds
/// leaves none open.
struct OpenWindows {
    /// Arrival Usage: closed in the order the devices opened them.
    open: Vec<(Client, ProfileWindow)>,
}

impl OpenWindows {
    fn open(clients: &[Client]) -> Result<Self, ProfileError> {
        let mut windows = Self { open: Vec::new() };
        for client in clients {
            let window = client.profile_start_at_launch()?;
            windows.open.push((client.clone(), window));
        }
        Ok(windows)
    }

    fn close(mut self) -> Result<Vec<ProfileDuration>, ProfileError> {
        let closed: Vec<_> = self
            .open
            .drain(..)
            .map(|(client, window)| client.profile_end(window))
            .collect();
        closed.into_iter().collect()
    }
}

impl Drop for OpenWindows {
    fn drop(&mut self) {
        for (client, window) in self.open.drain(..) {
            client.profile_abandon(window);
        }
    }
}

/// The slowest of `durations`, measured only where each of them is.
fn slowest(durations: Vec<ProfileDuration>) -> ProfileDuration {
    let method = durations[0].timing_method();
    let resolve = async move {
        // Resolved together: a profile submits its read only once polled.
        let ticks =
            futures_util::future::join_all(durations.into_iter().map(ProfileDuration::resolve))
                .await;
        let ticks: Option<Vec<ProfileTicks>> = ticks.into_iter().collect();
        ticks?.into_iter().max_by_key(ProfileTicks::duration)
    };
    ProfileDuration::new(Box::pin(resolve), method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use cubecl_common::profile::{Duration, Instant, TimingMethod};

    fn taking(micros: u64) -> ProfileDuration {
        let start = Instant::now();
        ProfileDuration::new_system_time(start, start + Duration::from_micros(micros))
    }

    fn resolved(durations: Vec<ProfileDuration>) -> Option<Duration> {
        cubecl_environment::future::block_on(slowest(durations).resolve())
            .map(|ticks| ticks.duration())
    }

    #[test]
    fn a_sample_is_the_slowest_devices() {
        assert_eq!(
            resolved(vec![taking(3), taking(7), taking(5)]),
            Some(Duration::from_micros(7))
        );
    }

    #[test]
    fn a_sample_one_device_did_not_measure_is_none() {
        let unmeasured = ProfileDuration::new(Box::pin(async { None }), TimingMethod::System);
        assert_eq!(resolved(vec![taking(3), unmeasured]), None);
    }

    #[test]
    #[should_panic(expected = "a tune runs on a device at least")]
    fn a_tune_on_no_device_panics() {
        CandidateProfiler::new(&[]);
    }
}
