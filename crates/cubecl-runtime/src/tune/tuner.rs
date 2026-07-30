#[cfg(std_io)]
use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;
use derive_more::Display;

use core::time::Duration;

use cubecl_environment::sync::Mutex;

use alloc::string::{String, ToString};
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};

use crate::config::Logger;
#[cfg(std_io)]
use crate::config::autotune::AutotuneLogLevel;
use crate::server::LaunchError;
use crate::tune::{AutotuneLoggerExt, AutotuneResult, TimeBound, TuneCache, tune_benchmark};
use crate::{client::ComputeClient, runtime::Runtime};
use cubecl_environment::config::RuntimeConfig;

use super::{
    AutotuneKey, AutotuneOutput, TunableSet, TuneCacheResult, TuneFn, TuneInputs, TunePlan,
};

#[derive(Debug)]
/// Runs autotune benchmarks for a single device and caches the results.
///
/// On wasm, [`tune`](Self::tune) spawns its work on the browser event loop; elsewhere
/// it blocks inline. Either way the benchmarking itself is synchronous; only the
/// per-sample profile resolution is awaited.
pub struct Tuner<K: AutotuneKey> {
    cache: Arc<Mutex<TuneCache<K>>>,
    logger: Arc<Mutex<Logger>>,
}

/// The measured outcome for a given autotune invocation.
#[cfg_attr(autotune_persistence, derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct AutotuneOutcome {
    /// The name of the tunable.
    pub name: String,
    /// The index of the tunable.
    pub index: usize,
    /// The computation benchmark results.
    pub computation: BenchmarkComputations,
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
#[cfg_attr(autotune_persistence, derive(serde::Serialize, serde::Deserialize))]
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
    /// Time spent launching, when steps are being logged. The samples are still unresolved at
    /// that point, so the resolution wait is added in [`process_request`] before it is reported.
    launch: Option<Duration>,
}

/// Everything a benchmarking strategy needs, prepared once by [`Tuner::check_tune`] and handed to
/// whichever strategy runs. `'t` borrows the tunable set, `'i` the benchmark inputs.
struct TuneJob<'t, 'i, K: AutotuneKey, F: TuneInputs, Out> {
    key: K,
    autotunables: Vec<&'t TuneFn<F, Out>>,
    test_inputs: <F as TuneInputs>::At<'i>,
    plan: TunePlan,
    results: Vec<AutotuneResult>,
    #[cfg(not(target_family = "wasm"))]
    limit: Option<Duration>,
    #[cfg(not(target_family = "wasm"))]
    short_circuit: bool,
    #[cfg(autotune_persistence)]
    checksum: String,
    log_context: Option<crate::tune::AutotuneLogContext>,
}

impl<K: AutotuneKey, F: TuneInputs, Out> TuneJob<'_, '_, K, F, Out> {
    fn into_request(self, pending: Vec<PendingBench>) -> TuneRequest<K> {
        TuneRequest {
            key: self.key,
            results: self.results,
            #[cfg(autotune_persistence)]
            checksum: self.checksum,
            log_context: self.log_context,
            pending,
        }
    }
}

/// A queued tuning job: all data needed to resolve samples and commit the result.
/// Holds no references so it's trivially `Send + 'static` for the wasm spawn path.
struct TuneRequest<K: AutotuneKey> {
    key: K,
    results: Vec<AutotuneResult>,
    #[cfg(autotune_persistence)]
    checksum: String,
    log_context: Option<crate::tune::AutotuneLogContext>,
    pending: Vec<PendingBench>,
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Create a tuner. Its cache is seeded from the persistent cache when
    /// persistence is available (disk on native, browser storage on wasm with
    /// the `browser-cache` feature).
    pub fn new(name: &str, device_id: &str) -> Self {
        Self {
            cache: Arc::new(Mutex::new(TuneCache::new(name, device_id))),
            logger: Arc::new(Mutex::new(Logger::new())),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    ///
    /// This resets the cache when the environment switched but does not
    /// re-hydrate it from persistence, so right after a switch it reports a
    /// [`Miss`](TuneCacheResult::Miss) even for keys the new environment has
    /// cached. It is a fast-path probe: a miss here is expected to fall through
    /// to [`check_tune`](Self::check_tune), which hydrates and resolves the
    /// real state. Don't rely on it as a standalone "is this cached?" query.
    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        #[cfg_attr(not(autotune_persistence), allow(unused_mut))]
        let mut cache = self.cache.lock();
        #[cfg(autotune_persistence)]
        cache.reset_if_environment_switched();

        cache.fastest(key)
    }

    /// Fetch the logger instance.
    pub fn logger(&self) -> Arc<Mutex<Logger>> {
        self.logger.clone()
    }

    /// Check the cache, validate checksums if needed, and kick off a tuning job if the
    /// key is a miss. Returns the resolved cache state.
    pub fn check_tune<'a, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out>,
        #[cfg_attr(not(autotune_persistence), allow(unused))] checksum: impl FnOnce() -> String
        + Send
        + Sync,
        client: &ComputeClient<R>,
        mut log_context: Option<crate::tune::AutotuneLogContext>,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        {
            let mut cache = self.cache.lock();
            #[cfg(autotune_persistence)]
            cache.reset_if_environment_switched();
            let cur = cache.fastest(key);

            // Browser hydration is asynchronous, so persistent entries may
            // have arrived after construction. Ingest them before starting a
            // redundant tune.
            #[cfg(autotune_persistence)]
            let cur = if matches!(cur, TuneCacheResult::Miss) {
                cache.sync_persistent();
                cache.fastest(key)
            } else {
                cur
            };

            #[cfg(autotune_persistence)]
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
            // path insert, `process_request`), and the mutex is non-reentrant.
        }

        log::info!("Tuning {key}");

        let autotunables = tunables.autotunables().collect::<Vec<_>>();
        let results: Vec<AutotuneResult> = autotunables
            .iter()
            .map(|a| {
                AutotuneResult::error(AutotuneError::Skip {
                    name: a.name.to_string(),
                })
            })
            .collect();

        #[cfg(autotune_persistence)]
        let checksum = tunables.compute_checksum();

        // Fast path: single tunable, no benchmarking needed.
        if results.len() == 1 {
            self.cache.lock().cache_insert(key.clone(), 0);
            return TuneCacheResult::Hit { fastest_index: 0 };
        }

        let test_inputs = tunables.generate_inputs(key, inputs);
        let plan = tunables.plan(key);
        let bounds = tunables.bounds(key, inputs);
        let limit = bounds.as_ref().and_then(|bounds| bounds.time_limit());

        log_context.set_bounds(bounds);
        log_context.set_limit(limit);

        // The slowest median duration still considered close enough to peak throughput.
        // Only used on native, where a benchmark can be resolved inline to exit early.
        #[cfg(not(target_family = "wasm"))]
        let short_circuit = limit.is_some()
            && tunables.is_short_circuit_enabled()
            && !crate::config::CubeClRuntimeConfig::get()
                .autotune
                .disable_short_circuit;

        let job = TuneJob {
            key: key.clone(),
            autotunables,
            test_inputs,
            plan,
            results,
            #[cfg(not(target_family = "wasm"))]
            limit,
            #[cfg(not(target_family = "wasm"))]
            short_circuit,
            #[cfg(autotune_persistence)]
            checksum,
            log_context,
        };

        #[cfg(not(target_family = "wasm"))]
        if crate::config::CubeClRuntimeConfig::get()
            .autotune
            .bench
            .adaptive
        {
            return self.tune_adaptive(job, client);
        }

        self.tune_fixed_samples(job, client)
    }

    /// Round robin the candidates, eliminating them as the evidence allows. Native only: the
    /// driver has to resolve samples between rounds, which it cannot do on the browser event loop.
    #[cfg(not(target_family = "wasm"))]
    fn tune_adaptive<'i, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        mut job: TuneJob<'_, 'i, K, F, Out>,
        client: &ComputeClient<R>,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'i>: Clone + Send,
    {
        let schedule = crate::tune::schedule::Schedule {
            config: crate::config::CubeClRuntimeConfig::get()
                .autotune
                .bench
                .clone(),
            limit: job.limit,
            short_circuit: job.short_circuit,
            track_steps: job.log_context.is_some(),
        };

        let (steps, short_circuit) = schedule.run_plan(
            &job.key,
            &mut job.plan,
            &job.autotunables,
            &job.test_inputs,
            client,
            &mut job.results,
        );

        for (name, duration) in steps {
            job.log_context.push_tuning_step(name, duration);
        }
        if let Some(name) = short_circuit {
            job.log_context.push_short_circuit(name);
        }

        cubecl_environment::future::block_on(process_request(
            job.into_request(Vec::new()),
            &self.cache,
            &self.logger,
        ))
    }

    /// Benchmark every candidate with a fixed sample count, resolving the samples afterwards.
    /// This is the only strategy available on wasm, where nothing can be awaited inline.
    fn tune_fixed_samples<'i, R: Runtime, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        mut job: TuneJob<'_, 'i, K, F, Out>,
        client: &ComputeClient<R>,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'i>: Clone + Send,
    {
        // The batch-retry check below reads this through `cfg!`, which keeps
        // the name alive on wasm too; the assignment is native-only, so it
        // simply stays false there.
        #[cfg(not(target_family = "wasm"))]
        let mut batch_success = false;
        #[cfg(target_family = "wasm")]
        let batch_success = false;

        // Walk the plan batch by batch, launching each benchmark synchronously. A
        // successful launch queues a `PendingBench` for the async resolver below;
        // launch errors go straight into `results`. Retry the next batch if a whole
        // batch failed to queue anything.
        let mut pending = Vec::<PendingBench>::new();
        loop {
            let tunable_indices = job.plan.next();

            if tunable_indices.is_empty() {
                let key = &job.key;
                panic!(
                    "Can't execute the autotune plan for key: {key:?}\n - plan: {:?}\n - results: {:?}",
                    job.plan, job.results
                );
            }

            for index in tunable_indices {
                let op = job.autotunables[index];

                let start_time = job
                    .log_context
                    .is_some()
                    .then(cubecl_common::profile::Instant::now);

                match tune_benchmark(op, job.test_inputs.clone(), client.clone()) {
                    Ok(profiles) => {
                        let bench = PendingBench {
                            index,
                            name: op.name.clone(),
                            profiles,
                            launch: start_time.map(|start| start.elapsed()),
                        };

                        #[cfg(not(target_family = "wasm"))]
                        if job.short_circuit {
                            let result = cubecl_environment::future::block_on(resolve_bench(bench));

                            // short_circuit is only true when limit.is_some() => unwrap is fine.
                            let close_enough = result
                                .outcome
                                .as_ref()
                                .is_ok_and(|out| out.computation.median <= job.limit.unwrap());

                            batch_success |= result.outcome.is_ok();
                            job.results[index] = result;

                            if let Some(start) = start_time {
                                job.log_context
                                    .push_tuning_step(op.name.to_string(), start.elapsed());
                            }

                            if close_enough {
                                job.log_context.push_short_circuit(op.name.to_string());
                                break;
                            }

                            continue;
                        }

                        // The step is reported once `process_request` has resolved the samples,
                        // so the logged duration covers benchmarking and not just the launch.
                        pending.push(bench);
                    }
                    Err(err) => {
                        job.results[index] = AutotuneResult::error(err);
                        if let Some(start) = start_time {
                            job.log_context
                                .push_tuning_step(op.name.to_string(), start.elapsed());
                        }
                    }
                }
            }

            #[cfg(not(target_family = "wasm"))]
            if !pending.is_empty() || batch_success {
                break;
            }
            #[cfg(target_family = "wasm")]
            if !pending.is_empty() {
                break;
            }
        }

        let request = job.into_request(pending);

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
        cubecl_environment::future::block_on(process_request(request, &self.cache, &self.logger))
    }
}

/// Await every sample of a single benchmark and fold them into one result.
///
/// The samples are resolved concurrently: a profile only submits its readback when
/// first polled, so awaiting them one by one would serialize a device round-trip per
/// sample.
async fn resolve_bench(bench: PendingBench) -> AutotuneResult {
    let PendingBench {
        index,
        name,
        profiles,
        launch: _,
    } = bench;

    let Some(first) = profiles.first() else {
        return AutotuneResult::error(AutotuneError::Unknown {
            name: name.to_string(),
            err: "No profiling available".to_string(),
        });
    };
    let timing_method = first.timing_method();

    let durations: Vec<Duration> =
        futures_util::future::join_all(profiles.into_iter().map(ProfileDuration::resolve))
            .await
            .into_iter()
            .map(|ticks| ticks.duration())
            .collect();

    AutotuneResult::success(AutotuneOutcome::new(
        name,
        index,
        BenchmarkComputations::new(&BenchmarkDurations::from_durations(
            timing_method,
            durations,
        )),
    ))
}

/// Await every profile sample, pick the fastest tunable, commit to the cache.
async fn process_request<K: AutotuneKey>(
    request: TuneRequest<K>,
    cache: &Mutex<TuneCache<K>>,
    logger: &Mutex<Logger>,
) -> TuneCacheResult {
    let TuneRequest {
        key,
        mut results,
        #[cfg(autotune_persistence)]
        checksum,
        mut log_context,
        pending,
    } = request;

    for bench in pending {
        let index = bench.index;
        let launch = bench.launch;
        let name = bench.name.clone();

        let started = cubecl_common::profile::Instant::now();
        let result = resolve_bench(bench).await;

        if let Some(launch) = launch {
            log_context.push_tuning_step(name, launch + started.elapsed());
        }

        results[index] = result;
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
        log_context.log_result(&mut logger.lock(), &key, &results);
        cache.lock().cache_insert(key.clone(), fastest_index);
        #[cfg(autotune_persistence)]
        cache
            .lock()
            .persistent_cache_insert(key, checksum, fastest_index, results);
    }

    TuneCacheResult::Hit { fastest_index }
}

#[cfg(feature = "autotune-checks")]
pub(crate) fn check_autotune_outputs<O: AutotuneOutput>(
    mut checks_outputs: Vec<(String, Result<O, AutotuneError>)>,
) -> Vec<crate::tune::log::CheckResult> {
    if checks_outputs.is_empty() {
        return Vec::new();
    }

    let reference_idx = checks_outputs
        .iter()
        .position(|(_, res)| res.is_ok())
        .unwrap_or(checks_outputs.len() - 1);
    let reference = checks_outputs.remove(reference_idx);
    let reference_result = reference.1;
    #[cfg(std_io)]
    let reference_name = reference.0;

    let is_recording = is_recording_enabled();

    #[cfg(std_io)]
    {
        let reference_passed = reference_result.is_ok();
        let mut check_results = execute_checks(checks_outputs, reference_result, is_recording);
        check_results.push(crate::tune::log::CheckResult {
            name: reference_name,
            passed: reference_passed,
        });

        check_results
    }

    #[cfg(not(std_io))]
    {
        execute_checks(checks_outputs, reference_result, is_recording)
    }
}

/// Whether a mismatch should be collected rather than fatal: it can only be reported if something
/// is recording the results, so with no recorder a failed check panics on the spot instead of
/// passing silently.
#[cfg(feature = "autotune-checks")]
fn is_recording_enabled() -> bool {
    crate::config::CubeClRuntimeConfig::get()
        .autotune
        .recording_enabled()
}

#[cfg(feature = "autotune-checks")]
fn execute_checks<O: AutotuneOutput>(
    checks_outputs: Vec<(String, Result<O, AutotuneError>)>,
    reference_result: Result<O, AutotuneError>,
    is_recording: bool,
) -> Vec<crate::tune::log::CheckResult> {
    let mut check_results = Vec::new();

    let Ok(reference) = reference_result else {
        for (name, _) in checks_outputs.into_iter() {
            check_results.push(crate::tune::log::CheckResult {
                name,
                passed: false,
            });
        }
        return check_results;
    };

    for (name, other_result) in checks_outputs.into_iter() {
        if let Ok(other) = other_result {
            let passed = check_equivalence(&reference, other, is_recording);
            check_results.push(crate::tune::log::CheckResult { name, passed });
        } else {
            check_results.push(crate::tune::log::CheckResult {
                name,
                passed: false,
            });
        }
    }

    check_results
}

#[cfg(feature = "autotune-checks")]
fn check_equivalence<O: AutotuneOutput>(reference: &O, other: O, is_recording: bool) -> bool {
    // When the results are being recorded, we catch the panic so we can collect and report every
    // check failure. With nothing recording, we let it panic immediately rather than pass silently.
    if is_recording {
        #[cfg(std_io)]
        {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                reference.check_equivalence(other);
            }))
            .is_ok()
        }
        #[cfg(not(std_io))]
        {
            reference.check_equivalence(other);
            true
        }
    } else {
        reference.check_equivalence(other);
        true
    }
}
