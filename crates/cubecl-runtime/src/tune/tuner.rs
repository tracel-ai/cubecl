#[cfg(not(target_family = "wasm"))]
use alloc::boxed::Box;
#[cfg(autotune_persistence)]
use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cubecl_common::profile::ProfileDuration;
use derive_more::Display;

use core::time::Duration;

use cubecl_environment::sync::Mutex;

use alloc::string::{String, ToString};
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};

use crate::client::Client;
use crate::config::Logger;
#[cfg(autotune_persistence)]
use crate::config::autotune::AutotuneLogLevel;
use crate::server::LaunchError;
use crate::tune::{AutotuneLoggerExt, AutotuneResult, TimeBound, TuneCache, tune_benchmark};
use cubecl_environment::config::RuntimeConfig;

#[cfg(not(target_family = "wasm"))]
use super::TuneFn;
use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneCacheResult, TuneInputs, TunePlan};

#[derive(Debug)]
/// Benchmarks and caches autotune candidates for one device.
pub struct Tuner<K: AutotuneKey> {
    cache: Arc<Mutex<TuneCache<K>>>,
    logger: Arc<Mutex<Logger>>,
    #[cfg(target_family = "wasm")]
    browser_tuning: Mutex<hashbrown::HashMap<K, Arc<Mutex<BrowserTuning<K>>>>>,
}

/// Returns the number of browser tuning rounds launched by this process.
#[cfg(target_family = "wasm")]
pub fn rounds_launched() -> usize {
    ROUNDS_LAUNCHED.load(core::sync::atomic::Ordering::Relaxed)
}

#[cfg(target_family = "wasm")]
static ROUNDS_LAUNCHED: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

#[cfg(target_family = "wasm")]
#[derive(Debug)]
struct BrowserTuning<K: AutotuneKey> {
    key: K,
    plan: TunePlan,
    results: Vec<AutotuneResult>,
    batch: Vec<usize>,
    in_flight: usize,
    resolved: Vec<(usize, Option<(String, Duration)>, AutotuneResult)>,
    settled: bool,
    #[cfg(autotune_persistence)]
    limit: Option<Duration>,
    #[cfg(autotune_persistence)]
    bounds: Option<crate::tune::Bounds>,
    #[cfg(autotune_persistence)]
    checksum: String,
    log_context: Option<crate::tune::AutotuneLogContext>,
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
    /// A sample did not produce a timing measurement.
    #[display("{name}: A profiled sample carried no measurement.")]
    NotMeasured {
        /// The name of the tunable.
        name: String,
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

struct PendingBench {
    index: usize,
    name: String,
    profiles: Vec<ProfileDuration>,
    launch: Option<Duration>,
}

#[cfg(not(target_family = "wasm"))]
struct TuneJob<'t, 'i, K: AutotuneKey, F: TuneInputs, Out> {
    key: K,
    autotunables: Vec<&'t TuneFn<F, Out>>,
    test_inputs: <F as TuneInputs>::At<'i>,
    evictor: Option<Box<crate::tune::Evictor<'i>>>,
    plan: TunePlan,
    results: Vec<AutotuneResult>,
    limit: Option<Duration>,
    #[cfg(autotune_persistence)]
    bounds: Option<crate::tune::Bounds>,
    short_circuit: bool,
    #[cfg(autotune_persistence)]
    checksum: String,
    log_context: Option<crate::tune::AutotuneLogContext>,
}

#[cfg(not(target_family = "wasm"))]
impl<K: AutotuneKey, F: TuneInputs, Out> TuneJob<'_, '_, K, F, Out> {
    fn into_request(self, pending: Vec<PendingBench>, decided: Option<usize>) -> TuneRequest<K> {
        TuneRequest {
            key: self.key,
            results: self.results,
            #[cfg(autotune_persistence)]
            checksum: self.checksum,
            log_context: self.log_context,
            pending,
            decided,
            #[cfg(autotune_persistence)]
            limit: self.limit,
            #[cfg(autotune_persistence)]
            bounds: self.bounds,
        }
    }
}

struct TuneRequest<K: AutotuneKey> {
    key: K,
    results: Vec<AutotuneResult>,
    #[cfg(autotune_persistence)]
    checksum: String,
    log_context: Option<crate::tune::AutotuneLogContext>,
    pending: Vec<PendingBench>,
    decided: Option<usize>,
    #[cfg(autotune_persistence)]
    limit: Option<Duration>,
    #[cfg(autotune_persistence)]
    bounds: Option<crate::tune::Bounds>,
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Creates a tuner backed by the configured cache.
    pub fn new(name: &str, device_id: &str) -> Self {
        Self {
            cache: Arc::new(Mutex::new(TuneCache::new(name, device_id))),
            logger: Arc::new(Mutex::new(Logger::new())),
            #[cfg(target_family = "wasm")]
            browser_tuning: Mutex::new(hashbrown::HashMap::new()),
        }
    }

    /// Returns the fastest cached candidate for a key.
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
    pub fn check_tune<'a, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        key: &K,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out>,
        #[cfg_attr(not(autotune_persistence), allow(unused))] checksum: impl FnOnce() -> String
        + Send
        + Sync,
        client: &Client,
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

            #[cfg(autotune_persistence)]
            let cur = if matches!(cur, TuneCacheResult::Miss) {
                cache.sync_persistent();
                cache.fastest(key)
            } else {
                cur
            };

            #[cfg(autotune_persistence)]
            if matches!(cur, TuneCacheResult::Miss) && !cache.hydrated() {
                return TuneCacheResult::Pending;
            }

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
                TuneCacheResult::Hit { .. } => return cur,
                TuneCacheResult::Pending => {
                    #[cfg(target_family = "wasm")]
                    {
                        let progress = self.browser_tuning.lock().get(key).cloned();
                        if let Some(progress) = progress {
                            drop(cache);
                            return self.advance(&progress, inputs, tunables, client);
                        }
                    }
                    return cur;
                }
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

        #[cfg(not(target_family = "wasm"))]
        let test_inputs = tunables.generate_inputs(key, inputs);
        let plan = tunables.plan(key);
        let bounds = tunables.bounds(key, inputs);
        let limit = bounds.as_ref().and_then(|bounds| bounds.time_limit());

        log_context.set_bounds(bounds.clone());
        log_context.set_limit(limit);

        // The slowest median duration still considered close enough to peak throughput.
        // Only used on native, where a benchmark can be resolved inline to exit early.
        #[cfg(not(target_family = "wasm"))]
        let short_circuit = limit.is_some()
            && tunables.is_short_circuit_enabled()
            && !crate::config::CubeClRuntimeConfig::get()
                .autotune
                .disable_short_circuit;

        #[cfg(target_family = "wasm")]
        {
            let _ = autotunables;
            let progress = Arc::new(Mutex::new(BrowserTuning {
                key: key.clone(),
                plan,
                results,
                batch: Vec::new(),
                in_flight: 0,
                resolved: Vec::new(),
                settled: false,
                #[cfg(autotune_persistence)]
                limit,
                #[cfg(autotune_persistence)]
                bounds,
                #[cfg(autotune_persistence)]
                checksum,
                log_context,
            }));
            self.browser_tuning
                .lock()
                .insert(key.clone(), progress.clone());
            return self.advance(&progress, inputs, tunables, client);
        }

        #[cfg(not(target_family = "wasm"))]
        {
            let job = TuneJob {
                key: key.clone(),
                autotunables,
                test_inputs,
                evictor: tunables.evictor(key, inputs),
                plan,
                results,
                limit,
                #[cfg(autotune_persistence)]
                bounds,
                short_circuit,
                #[cfg(autotune_persistence)]
                checksum,
                log_context,
            };

            if crate::config::CubeClRuntimeConfig::get()
                .autotune
                .bench
                .adaptive
            {
                return self.tune_adaptive(job, client);
            }

            self.tune_fixed_samples(job, client)
        }
    }

    /// Returns the fastest candidate measured by an in-progress tune.
    pub fn provisional(&self, key: &K) -> Option<usize> {
        #[cfg(target_family = "wasm")]
        {
            let progress = self.browser_tuning.lock().get(key).cloned()?;
            let state = progress.lock();
            state
                .results
                .iter()
                .chain(state.resolved.iter().map(|(_, _, result)| result))
                .filter_map(|result| result.outcome.as_ref().ok())
                .min_by_key(|outcome| outcome.computation.score())
                .map(|outcome| outcome.index)
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = key;
            None
        }
    }

    #[cfg(target_family = "wasm")]
    fn advance<'a, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        progress: &Arc<Mutex<BrowserTuning<K>>>,
        inputs: &F::At<'a>,
        tunables: &TunableSet<K, F, Out>,
        client: &Client,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'a>: Clone + Send,
    {
        let mut state = progress.lock();
        if state.settled || state.in_flight > 0 {
            return TuneCacheResult::Pending;
        }
        for (index, step, result) in state.resolved.drain(..).collect::<Vec<_>>() {
            if let Some((name, duration)) = step {
                state.log_context.push_tuning_step(name, duration);
            }
            state.results[index] = result;
        }

        let bench = crate::config::CubeClRuntimeConfig::get()
            .autotune
            .bench
            .clone();
        let launches = (bench.warmup_samples.max(1) + bench.samples().1) as u32;
        let medians = state
            .results
            .iter()
            .filter_map(|result| result.outcome.as_ref().ok())
            .map(|outcome| outcome.computation.median);
        let spent: Duration = medians.clone().map(|median| median * launches).sum();
        let slowest = medians.max().unwrap_or(Duration::ZERO);
        let measured = spent > Duration::ZERO;
        let over_budget = measured && bench.browser_budget().is_some_and(|budget| spent > budget);
        let round_size = if slowest.is_zero() {
            1
        } else {
            (bench.browser_round().as_nanos() / (slowest * launches).as_nanos().max(1)).clamp(1, 64)
                as usize
        };

        let autotunables = tunables.autotunables().collect::<Vec<_>>();
        let mut evictor = tunables.evictor(&state.key, inputs);
        let mut pending = Vec::<PendingBench>::new();
        while pending.is_empty() && !over_budget {
            if state.batch.is_empty() {
                let next = state.plan.next();
                if next.is_empty() {
                    break;
                }
                state.batch = next;
            }
            let take = round_size.min(state.batch.len());
            let round: Vec<usize> = state.batch.drain(..take).collect();
            ROUNDS_LAUNCHED.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            for index in round {
                let op = autotunables[index];
                let start_time = state
                    .log_context
                    .is_some()
                    .then(cubecl_common::profile::Instant::now);
                match tune_benchmark(
                    op,
                    tunables.generate_inputs(&state.key, inputs),
                    client.clone(),
                    evictor.as_deref_mut(),
                ) {
                    Ok(profiles) => pending.push(PendingBench {
                        index,
                        name: op.name.clone(),
                        profiles,
                        launch: start_time.map(|start| start.elapsed()),
                    }),
                    Err(err) => {
                        state.results[index] = AutotuneResult::error(err);
                        if let Some(start) = start_time {
                            state
                                .log_context
                                .push_tuning_step(op.name.to_string(), start.elapsed());
                        }
                    }
                }
            }
        }

        if pending.is_empty() {
            if !measured && !state.results.iter().any(|result| result.outcome.is_ok()) {
                let key = &state.key;
                panic!(
                    "Can't execute the autotune plan for key: {key:?}\n - plan: {:?}\n - results: {:?}",
                    state.plan, state.results
                );
            }
            state.settled = true;
            self.browser_tuning.lock().remove(&state.key);
            let request = TuneRequest {
                key: state.key.clone(),
                results: core::mem::take(&mut state.results),
                #[cfg(autotune_persistence)]
                checksum: state.checksum.clone(),
                log_context: state.log_context.take(),
                pending: Vec::new(),
                decided: None,
                #[cfg(autotune_persistence)]
                limit: state.limit,
                #[cfg(autotune_persistence)]
                bounds: state.bounds.clone(),
            };
            drop(state);
            let cache = self.cache.clone();
            let logger = self.logger.clone();
            wasm_bindgen_futures::spawn_local(async move {
                process_request(request, &cache, &logger).await;
            });
            return TuneCacheResult::Pending;
        }

        state.in_flight = pending.len();
        drop(state);
        let progress = progress.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let resolved = futures_util::future::join_all(pending.into_iter().map(|bench| {
                let index = bench.index;
                let name = bench.name.clone();
                let launch = bench.launch;
                async move {
                    let started = cubecl_common::profile::Instant::now();
                    let result = resolve_bench(bench).await;
                    let step = launch.map(|launch| (name, launch + started.elapsed()));
                    (index, step, result)
                }
            }))
            .await;
            let mut state = progress.lock();
            state.in_flight -= resolved.len();
            state.resolved.extend(resolved);
        });
        TuneCacheResult::Pending
    }

    /// Round robin the candidates, eliminating them as the evidence allows. Native only: the
    /// driver has to resolve samples between rounds, which it cannot do on the browser event loop.
    #[cfg(not(target_family = "wasm"))]
    fn tune_adaptive<'i, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        mut job: TuneJob<'_, 'i, K, F, Out>,
        client: &Client,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'i>: Clone + Send,
    {
        let mut schedule = crate::tune::schedule::Schedule {
            config: crate::config::CubeClRuntimeConfig::get()
                .autotune
                .bench
                .clone(),
            limit: job.limit,
            short_circuit: job.short_circuit,
            track_steps: job.log_context.is_some(),
            evictor: job.evictor.take(),
        };

        let outcome = schedule.run_plan(
            &job.key,
            &mut job.plan,
            &job.autotunables,
            &job.test_inputs,
            client,
            &mut job.results,
        );

        for (name, duration) in outcome.steps {
            job.log_context.push_tuning_step(name, duration);
        }
        if let Some(name) = outcome.short_circuit {
            job.log_context.push_short_circuit(name);
        }

        let request = job.into_request(Vec::new(), outcome.decided);

        cubecl_environment::future::block_on(process_request(request, &self.cache, &self.logger))
    }

    /// Benchmark every candidate with a fixed sample count, resolving the samples afterwards.
    #[cfg(not(target_family = "wasm"))]
    fn tune_fixed_samples<'i, F: TuneInputs, Out: AutotuneOutput>(
        &self,
        mut job: TuneJob<'_, 'i, K, F, Out>,
        client: &Client,
    ) -> TuneCacheResult
    where
        <F as TuneInputs>::At<'i>: Clone + Send,
    {
        let mut batch_success = false;

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

                match tune_benchmark(
                    op,
                    job.test_inputs.clone(),
                    client.clone(),
                    job.evictor.as_deref_mut(),
                ) {
                    Ok(profiles) => {
                        let bench = PendingBench {
                            index,
                            name: op.name.clone(),
                            profiles,
                            launch: start_time.map(|start| start.elapsed()),
                        };

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

            if !pending.is_empty() || batch_success {
                break;
            }
        }

        // Every candidate here carries the same sample count, so scoring them against each other
        // is a fair comparison and `process_request` can make the call.
        let request = job.into_request(pending, None);

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

    // One unmeasured sample disqualifies the candidate, the way one failed
    // sample does. Dropping it and averaging the rest would let a candidate
    // whose window went untimed be judged on its remaining runs.
    let Some(durations) =
        futures_util::future::join_all(profiles.into_iter().map(ProfileDuration::resolve))
            .await
            .into_iter()
            .map(|ticks| ticks.map(|ticks| ticks.duration()))
            .collect::<Option<Vec<Duration>>>()
    else {
        return AutotuneResult::error(AutotuneError::NotMeasured {
            name: name.to_string(),
        });
    };

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
        decided,
        #[cfg(autotune_persistence)]
        limit,
        #[cfg(autotune_persistence)]
        bounds,
    } = request;

    // Resolved concurrently, and each benchmark timed individually rather than timing the loop:
    // the profiles were all queued before any of them was polled, so awaiting them in turn would
    // charge the first benchmark for draining the whole device queue and report the rest as
    // free.
    let resolved = futures_util::future::join_all(pending.into_iter().map(|bench| {
        let index = bench.index;
        let name = bench.name.clone();
        let launch = bench.launch;

        async move {
            let started = cubecl_common::profile::Instant::now();
            let result = resolve_bench(bench).await;
            let step = launch.map(|launch| (name, launch + started.elapsed()));

            (index, step, result)
        }
    }))
    .await;

    for (index, step, result) in resolved {
        if let Some((name, duration)) = step {
            log_context.push_tuning_step(name, duration);
        }

        results[index] = result;
    }

    // Read before the sort, which reorders `results` out of tunable order. A
    // decided candidate whose own outcome is an error is one `Schedule::run_plan`
    // picked with nothing measured — the tune executed but could not be timed.
    #[cfg(autotune_persistence)]
    let unmeasured = decided.is_some_and(|index| results[index].outcome.is_err());

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

    // The sort above orders what gets logged and persisted. It does not pick the winner when the
    // strategy already did: a scheduler that eliminates candidates leaves results built from
    // different sample counts behind, and `score` reads a short sample set as a stable one.
    let fastest_index = match decided {
        Some(index) => index,
        None => {
            results
                .first()
                .expect("At least one kernel needed.")
                .outcome
                .as_ref()
                .expect("At least one kernel has to succeed.")
                .index
        }
    };

    {
        log_context.log_result(&mut logger.lock(), &key, &results);
        // In-memory regardless: without it this key re-tunes on every call, and
        // a tune that measured nothing would keep failing the same way.
        cache.lock().cache_insert(key.clone(), fastest_index);

        // Not on disk, though. An unmeasured decision is a guess made to keep
        // the device thread alive, and the failures that produce one — a
        // profiling hiccup, timestamp query sets on a busy stream — are
        // transient. Persisting it would freeze the guess into every later
        // process and never measure the key again; letting it expire with this
        // one costs a re-tune and buys a real measurement.
        #[cfg(autotune_persistence)]
        if !unmeasured {
            cache.lock().persistent_cache_insert(
                key,
                checksum,
                crate::tune::PersistentCacheValue {
                    fastest_index,
                    results,
                    bounds,
                    limit,
                },
            );
        }
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
