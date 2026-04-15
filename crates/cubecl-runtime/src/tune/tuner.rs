use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;

use core::time::Duration;

use alloc::string::{String, ToString};
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};

use crate::config::{Logger, autotune::AutotuneLogLevel};
use crate::server::LaunchError;
use crate::tune::{AutotuneResult, TuneBenchmark, TuneCache};
use crate::{client::ComputeClient, runtime::Runtime};

use super::{AutotuneKey, AutotuneOutput, TuneInputs, TunableSet, TuneCacheResult};

#[derive(Debug)]
/// Executes autotune benchmarking and caching.
///
/// The only state is the cache and a logger, both shared with the short-lived future that
/// resolves a single tuning request. [`tune`](Self::tune) builds a [`TuneRequest`] and either
/// `spawn_local`s it on wasm or `block_on`s it inline on every other target — no long-running
/// worker, no request queue. The wasm and non-wasm paths are identical except for the driver.
pub struct Tuner<K: AutotuneKey> {
    cache: Arc<spin::Mutex<TuneCache<K>>>,
    logger: Arc<spin::Mutex<Logger>>,
}

/// The measured outcome for a given autotune invocation.
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct AutotuneOutcome {
    name: String,
    index: usize,
    computation: BenchmarkComputations,
}

impl core::fmt::Display for AutotuneOutcome {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Autotune[{}] name {} => {:?}",
            self.index, self.name, self.computation
        )
    }
}

/// Error from running autotune.
#[derive(Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneError {
    /// An unknown error happened.
    Unknown {
        /// The name of the tunable.
        name: String,
        /// The unknown error,
        err: String,
    },
    /// All samples are invalid.
    InvalidSamples {
        /// The name of the tunable.
        name: String,
    },
    /// No autotune was flagged as valid for the problem.
    ///
    /// # Warning
    ///
    /// This is an unrecoverable error and will cause a panic.
    NoValidKernelFound {
        /// The formatted context on why no valid kernel was found.
        context: String,
    },
    /// The autotune is skipped manually.
    Skip {
        /// The name of the skipped kernel.
        name: String,
    },

    /// An error happened when launching a kernel.
    Launch(LaunchError),
}

impl From<LaunchError> for AutotuneError {
    fn from(value: LaunchError) -> Self {
        Self::Launch(value)
    }
}

/// A successfully-queued benchmark: the profile futures for each sample, plus its metadata.
struct PendingBench {
    index: usize,
    name: String,
    profiles: Vec<ProfileDuration>,
}

/// A unit of work for the tuning worker. Holds everything needed to resolve a tuning job and
/// commit its result; deliberately carries no references so it is trivially `Send + 'static`.
struct TuneRequest<K: AutotuneKey> {
    key: K,
    results: Vec<AutotuneResult>,
    #[cfg(std_io)]
    checksum: String,
    context_logs: Option<String>,
    pending: Vec<PendingBench>,
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Returns a tuner with cache initialized from persistent cache. On wasm this spawns a
    /// Returns a tuner with cache initialized from persistent cache.
    pub fn new(name: &str, device_id: &str) -> Self {
        Self {
            cache: Arc::new(spin::Mutex::new(TuneCache::new(name, device_id))),
            logger: Arc::new(spin::Mutex::new(Logger::new())),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        self.cache.lock().fastest(key)
    }

    /// Fetch the fastest autotune operation index for an autotune key and validate the checksum.
    #[cfg(std_io)]
    pub fn validate_checksum(&self, key: &K, checksum: &str) {
        let mut log = self.logger.lock();
        if let AutotuneLogLevel::Full = log.log_level_autotune() {
            log.log_autotune(&format!("validate checksum key={key}, checksum={checksum}"));
        }
        self.cache.lock().validate_checksum(key, checksum)
    }

    /// Kick off (or attach to) a tuning job for `key`.
    ///
    /// Atomically checks the cache under the cache mutex:
    ///
    /// - **Hit**: another thread already finished the tune — returns an already-closed
    ///   receiver.
    /// - **Pending**: another thread is currently tuning, returns a clone of that job's
    ///   receiver.
    /// - **Miss / Unchecked**: claims the key as `Pending`, synchronously queues every
    ///   benchmark, hands the request to the worker, and returns the fresh receiver.
    pub fn tune<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        key: K,
        inputs: F::At<'a>,
        tunables: &TunableSet<K, F, Out>,
        client: &ComputeClient<R>,
    ) where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        {
            let mut cache = self.cache.lock();
            match cache.fastest(&key) {
                TuneCacheResult::Hit { .. } | TuneCacheResult::Pending => return,
                TuneCacheResult::Miss | TuneCacheResult::Unchecked => {
                    cache.mark_pending(key.clone())
                }
            }
            // Drop the cache guard here — the rest of this function needs to call
            // `self.cache.lock()` in a few places (fast-path insert, `process_request`).
            // `spin::Mutex` is non-reentrant, so holding this guard any longer would
            // self-deadlock the current thread.
        }

        log::info!("Tuning {key}");

        let autotunables = tunables.autotunables();
        let mut results: Vec<AutotuneResult> = autotunables
            .iter()
            .map(|a| {
                AutotuneResult::error(AutotuneError::Skip {
                    name: a.name().to_string(),
                })
            })
            .collect();

        #[cfg(std_io)]
        let checksum = tunables.compute_checksum();

        // Fast path: single tunable, no benchmarking needed.
        if autotunables.len() == 1 {
            self.cache.lock().cache_insert(key, 0);
            return;
        }

        let test_inputs = tunables.generate_inputs(&key, &inputs);
        let mut plan = tunables.plan(&key);
        let mut context_logs = match self.logger.lock().log_level_autotune() {
            AutotuneLogLevel::Full => Some(String::new()),
            _ => None,
        };

        // Walk the plan, synchronously launching every benchmark it picks. Each successful
        // queueing yields a `PendingBench` (profile futures + metadata) for the worker to
        // resolve later. Kernel-launch errors are recorded directly into `results`; if a whole
        // batch fails to queue anything, ask the plan for the next one and retry.
        let mut pending = Vec::<PendingBench>::new();
        loop {
            let tunable_indices = plan.next(context_logs.as_mut());

            if tunable_indices.is_empty() {
                panic!(
                    "Can't execute the autotune plan for key: {key:?}\n - plan: {plan:?}\n - results: {results:?}"
                );
            }

            for index in tunable_indices {
                let op = &autotunables[index];
                let name = op.name().to_string();
                let bench = TuneBenchmark::new(op.clone(), test_inputs.clone(), client.clone());
                match bench.profile() {
                    Ok(profiles) => pending.push(PendingBench {
                        index,
                        name,
                        profiles,
                    }),
                    Err(err) => {
                        results[index] = AutotuneResult::error(err);
                    }
                }
            }

            if !pending.is_empty() {
                break;
            }
        }

        let request = TuneRequest {
            key,
            results,
            #[cfg(std_io)]
            checksum,
            context_logs,
            pending,
        };

        // Drive the request. On wasm we spawn a minimal one-shot future that resolves it
        // cooperatively on the browser event loop; on every other target we `block_on` the
        // same future inline. Either way `process_request` is the one thing that actually
        // runs — the only difference is who polls it.
        #[cfg(target_family = "wasm")]
        {
            let cache = self.cache.clone();
            let logger = self.logger.clone();
            wasm_bindgen_futures::spawn_local(async move {
                process_request(request, &cache, &logger).await;
            });
        }

        #[cfg(not(target_family = "wasm"))]
        {
            cubecl_common::future::block_on(process_request(request, &self.cache, &self.logger));
        }
    }
}

/// Resolve a single [`TuneRequest`]: await every profile future, pick the fastest tunable,
/// commit the result to the cache.
async fn process_request<K: AutotuneKey>(
    request: TuneRequest<K>,
    cache: &spin::Mutex<TuneCache<K>>,
    logger: &spin::Mutex<Logger>,
) {
    let TuneRequest {
        key,
        mut results,
        #[cfg(std_io)]
        checksum,
        context_logs,
        pending,
    } = request;

    for bench in pending {
        let PendingBench {
            index,
            name,
            profiles,
        } = bench;

        if profiles.is_empty() {
            results[index] = AutotuneResult::error(AutotuneError::Unknown {
                name,
                err: "No profiling available".to_string(),
            });
            continue;
        }

        let timing_method = profiles.first().unwrap().timing_method();
        let mut durations = Vec::with_capacity(profiles.len());
        for profile in profiles {
            durations.push(profile.resolve().await.duration());
        }

        results[index] = AutotuneResult::success(AutotuneOutcome::new(
            name,
            index,
            BenchmarkComputations::new(&BenchmarkDurations::from_durations(
                timing_method,
                durations,
            )),
        ));
    }

    results.sort_by(|a, b| {
        let a = a
            .outcome
            .as_ref()
            .map(|r| r.computation.score())
            .unwrap_or(u64::MAX);
        let b = b
            .outcome
            .as_ref()
            .map(|r| r.computation.score())
            .unwrap_or(u64::MAX);
        a.cmp(&b)
    });

    let fastest_index = results
        .first()
        .expect("At least one kernel needed.")
        .outcome
        .as_ref()
        .expect("At least one kernel has to succeed.")
        .index;

    {
        log_result(&mut logger.lock(), &key, &results, context_logs.as_deref());
        cache.lock().cache_insert(key.clone(), fastest_index);
        #[cfg(std_io)]
        cache
            .lock()
            .persistent_cache_insert(key, checksum, fastest_index, results);
    }
}

/// Emit the autotune result through the logger at the currently configured level.
fn log_result<K: AutotuneKey>(
    logger: &mut Logger,
    key: &K,
    results: &[AutotuneResult],
    context_logs: Option<&str>,
) {
    match logger.log_level_autotune() {
        AutotuneLogLevel::Minimal => {
            let top_times = results
                .iter()
                .map(|r| {
                    let time = r
                        .outcome
                        .as_ref()
                        .map(|r| r.computation.median)
                        .unwrap_or(Duration::MAX);

                    let index = r.outcome.as_ref().map(|r| r.index).unwrap_or_default();
                    (index, time)
                })
                .take(3)
                .collect::<Vec<_>>();

            let result = results
                .first()
                .expect("At least one kernel needed.")
                .outcome
                .as_ref()
                .expect("At least one kernel has to succeed.");

            let context = context_logs.unwrap_or("");
            logger.log_autotune(&format!(
                "Fastest result {}-{key}. \n Top 3 times: {top_times:?}, context: {context}",
                result.name,
            ));
        }
        AutotuneLogLevel::Full => {
            let result = results
                .first()
                .expect("At least one kernel needed.")
                .outcome
                .as_ref()
                .expect("At least one kernel has to succeed.");

            let context = context_logs.unwrap_or("");
            logger.log_autotune(&format!(
                "Fastest result {}-{key}. Context: {context}",
                result.name,
            ));

            for result in results.iter() {
                match &result.outcome {
                    Ok(val) => {
                        logger.log_autotune(&format!("{val}"));
                    }
                    Err(err) => logger.log_autotune(&format!("{err:?}")),
                }
            }
        }
        AutotuneLogLevel::Disabled => {}
    }
}

#[cfg(feature = "autotune-checks")]
pub(crate) fn check_autotune_outputs<O: AutotuneOutput>(
    mut checks_outputs: Vec<Result<O, AutotuneError>>,
) {
    let reference = checks_outputs.remove(checks_outputs.len() - 1);

    if let Ok(reference) = reference {
        for other in checks_outputs.into_iter().flatten() {
            reference.check_equivalence(other);
        }
    }
}
