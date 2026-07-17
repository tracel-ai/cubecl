use core::sync::atomic::{AtomicU8, Ordering};

/// How the current stream is derived when no explicit override is active.
///
/// Loaded from the `[streaming]` section of `cubecl.toml` by `cubecl-runtime`,
/// or set programmatically with [`set_policy`]. An explicit [`set_policy`] call
/// always wins over the configuration file.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum StreamPolicy {
    /// One stream per OS thread (the historical behavior and the default).
    ///
    /// Correct for applications that submit work from plain threads. Under an
    /// async executor with work stealing, a logical task hopping between
    /// worker threads changes streams, causing spurious synchronization.
    #[default]
    PerThread,
    /// One stream per logical task.
    ///
    /// With the `tokio` feature enabled, a task keeps a stable stream across
    /// `.await` points even when the executor moves it between worker
    /// threads. Outside a task, or without the `tokio` feature, this behaves
    /// like [`PerThread`](StreamPolicy::PerThread).
    PerTask,
    /// Everything runs on stream `0`.
    ///
    /// The effective behavior on wasm and no-std targets today.
    Single,
}

const SET_BY_USER: u8 = 0b1000_0000;

static POLICY: AtomicU8 = AtomicU8::new(0);

fn encode(policy: StreamPolicy) -> u8 {
    match policy {
        StreamPolicy::PerThread => 0,
        StreamPolicy::PerTask => 1,
        StreamPolicy::Single => 2,
    }
}

fn decode(bits: u8) -> StreamPolicy {
    match bits & !SET_BY_USER {
        1 => StreamPolicy::PerTask,
        2 => StreamPolicy::Single,
        _ => StreamPolicy::PerThread,
    }
}

/// Sets the active stream policy.
///
/// Takes precedence over any policy loaded from configuration files, no matter
/// the call order. Should be called before the first kernel submissions:
/// already-resolved stream ids are not revisited.
pub fn set_policy(policy: StreamPolicy) {
    POLICY.store(encode(policy) | SET_BY_USER, Ordering::Relaxed);
}

/// Returns the active stream policy.
pub fn policy() -> StreamPolicy {
    decode(POLICY.load(Ordering::Relaxed))
}

/// Sets the stream policy from a configuration file.
///
/// A no-op if [`set_policy`] was already called: the user's explicit choice
/// wins over configuration.
#[doc(hidden)]
pub fn set_policy_from_config(policy: StreamPolicy) {
    let _ = POLICY.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        if current & SET_BY_USER != 0 {
            None
        } else {
            Some(encode(policy))
        }
    });
}

/// Resets the process-global policy so policy-mutating tests don't leak state.
#[cfg(test)]
pub(crate) fn tests_reset_policy() {
    POLICY.store(0, Ordering::Relaxed);
}

/// Serializes tests that read or mutate the process-global policy.
#[cfg(all(test, feature = "std"))]
pub(crate) fn tests_policy_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_policy_wins_over_config() {
        let _guard = tests_policy_lock();

        set_policy_from_config(StreamPolicy::Single);
        assert_eq!(policy(), StreamPolicy::Single);

        set_policy(StreamPolicy::PerTask);
        set_policy_from_config(StreamPolicy::Single);
        assert_eq!(policy(), StreamPolicy::PerTask);

        tests_reset_policy();
    }
}
