use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;
use derive_more::Display;

use core::time::Duration;

use alloc::string::{String, ToString};
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};

use crate::config::{Logger, autotune::AutotuneLogLevel};
use crate::server::LaunchError;
use crate::tune::{AutotuneResult, TuneCache, tune_benchmark};
use crate::{client::ComputeClient, runtime::Runtime};

use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneCacheResult, TuneInputs};

#[derive(Debug)]
/// Runs autotune benchmarks for a single device and caches the results.
///
/// On wasm, [`tune`](Self::tune) spawns its work on the browser event loop; elsewhere
/// it blocks inline. Either way the benchmarking itself is synchronous; only the
/// per-sample profile resolution is awaited.
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
#[derive(Clone, Display)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneError {
    /// An unknown error happened.
    #[display("{name}: An unknown error happened.\n{err}")]
    Unknown {
        /// The name of the tunable.
        name: String,
        /// The unknown error,
        err: String,
    },
    /// All samples are invalid.
    #[display("{name}: All samples are invalid.")]
    InvalidSamples {
        /// The name of the tunable.
        name: String,
    },
    /// No autotune was flagged as valid for the problem.
    ///
    /// # Warning
    ///
    /// This is an unrecoverable error and will cause a panic.
    #[display("No autotune was flagged as valid for the problem.\n{context}")]
    NoValidKernelFound {
        /// The formatted context on why no valid kernel was found.
        context: String,
    },
    /// The autotune is skipped manually.
    #[display("{name}: The autotune is skipped manually.")]
    Skip {
        /// The name of the skipped kernel.
        name: String,
    },

    /// An error happened when launching a kernel.
    Launch(LaunchError),
}

impl core::fmt::Debug for AutotuneError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
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

/// A queued tuning job: all data needed to resolve samples and commit the result.
/// Holds no references so it's trivially `Send + 'static` for the wasm spawn path.
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
    /// Create a tuner. Its cache is seeded from the persistent on-disk cache when
    /// `std_io` is enabled.
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

    /// Check the cache, validate checksums if needed, and kick off a tuning job if the
    /// key is a miss. Returns the resolved cache state.
    pub fn check_tune<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out>,
        #[cfg_attr(not(std_io), allow(unused))] checksum: impl FnOnce() -> String + Send + Sync,
        client: &ComputeClient<R>,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        {
            let mut cache = self.cache.lock();
            let cur = cache.fastest(key);

            #[cfg(std_io)]
            let cur = if matches!(cur, TuneCacheResult::Unchecked) {
                let mut log = self.logger.lock();
                let checksum = checksum();
                if let AutotuneLogLevel::Full = log.log_level_autotune() {
                    log.log_autotune(&format!("validate checksum key={key}, checksum={checksum}"));
                }
                cache.validate_checksum(key, &checksum)
            } else {
                cur
            };

            match cur {
                TuneCacheResult::Hit { .. } | TuneCacheResult::Pending => return cur,
                TuneCacheResult::Miss | TuneCacheResult::Unchecked => {
                    cache.mark_pending(key.clone())
                }
            }
            // Scope the guard: the rest of this function re-locks `self.cache` (fast
            // path insert, `process_request`), and `spin::Mutex` is non-reentrant.
        }

        log::info!("Tuning {key}");

        let autotunables = tunables.autotunables().collect::<Vec<_>>();
        let mut results: Vec<AutotuneResult> = autotunables
            .iter()
            .map(|a| {
                AutotuneResult::error(AutotuneError::Skip {
                    name: a.name.to_string(),
                })
            })
            .collect();

        #[cfg(std_io)]
        let checksum = tunables.compute_checksum();

        // Fast path: single tunable, no benchmarking needed.
        if results.len() == 1 {
            self.cache.lock().cache_insert(key.clone(), 0);
            return TuneCacheResult::Hit { fastest_index: 0 };
        }

        let test_inputs = tunables.generate_inputs(key, inputs);
        let mut plan = tunables.plan(key);
        let mut context_logs = match self.logger.lock().log_level_autotune() {
            AutotuneLogLevel::Full => Some(String::new()),
            _ => None,
        };

        // Walk the plan batch by batch, launching each benchmark synchronously. A
        // successful launch queues a `PendingBench` for the async resolver below;
        // launch errors go straight into `results`. Retry the next batch if a whole
        // batch failed to queue anything.
        let mut pending = Vec::<PendingBench>::new();
        loop {
            let tunable_indices = plan.next(context_logs.as_mut());

            if tunable_indices.is_empty() {
                panic!(
                    "Can't execute the autotune plan for key: {key:?}\n - plan: {plan:?}\n - results: {results:?}"
                );
            }

            for index in tunable_indices {
                let op = autotunables[index];

                match tune_benchmark(op, test_inputs.clone(), client.clone()) {
                    Ok(profiles) => pending.push(PendingBench {
                        index,
                        name: op.name.clone(),
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
            key: key.clone(),
            results,
            #[cfg(std_io)]
            checksum,
            context_logs,
            pending,
        };

        // Resolve samples and commit the result. On wasm this runs on the browser
        // event loop; elsewhere it blocks inline.
        #[cfg(target_family = "wasm")]
        {
            let cache = self.cache.clone();
            let logger = self.logger.clone();
            wasm_bindgen_futures::spawn_local(async move {
                process_request(request, &cache, &logger).await;
            });

            return TuneCacheResult::Pending;
        }

        #[cfg(not(target_family = "wasm"))]
        cubecl_common::future::block_on(process_request(request, &self.cache, &self.logger))
    }
}

/// Await every profile sample, pick the fastest tunable, commit to the cache.
async fn process_request<K: AutotuneKey>(
    request: TuneRequest<K>,
    cache: &spin::Mutex<TuneCache<K>>,
    logger: &spin::Mutex<Logger>,
) -> TuneCacheResult {
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
                name: name.to_string(),
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
            name.to_string(),
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

    TuneCacheResult::Hit { fastest_index }
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
                    Err(err) => logger.log_autotune(&format!("{err}")),
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
