use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use async_channel::{Receiver, Sender};
use cubecl_common::profile::ProfileDuration;
use hashbrown::HashSet;

use core::time::Duration;

use alloc::string::{String, ToString};
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::config::{Logger, autotune::AutotuneLogLevel};
use crate::server::ComputeServer;
use crate::tune::{TuneBenchmark, TuneCache};

use super::{AutotuneKey, AutotuneOutput, TunableSet, TuneCacheResult, TuneFn, TunePlan};

#[derive(Debug)]
/// Executes autotune benchmarking and caching
pub struct Tuner<K: AutotuneKey> {
    tune_cache: TuneCache<K>,
    logger: Logger,
    channel: (Sender<AutotuneMessage<K>>, Receiver<AutotuneMessage<K>>),
    pub(crate) autotuning: HashSet<K>,
}

/// The measured outcome for a given autotune invocation.
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize, PartialEq, Eq))]
#[derive(new, Debug, Clone)]
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

enum AutotuneMessage<K> {
    Done {
        key: K,
        fastest_index: usize,
        results: Vec<Result<AutotuneOutcome, AutotuneError>>,
        #[cfg(std_io)]
        checksum: String,
    },
    #[allow(dead_code)]
    Pending(K),
}

/// Error from running autotune.
#[derive(Debug, PartialEq, Eq, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum AutotuneError {
    /// An unknown error happened.
    Unknown(String),
    /// All samples are invalid.
    InvalidSamples,
    /// The autotune is skipped manually.
    Skip,
}

impl From<String> for AutotuneError {
    fn from(value: String) -> Self {
        Self::Unknown(value)
    }
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Returns a tuner with cache initialized from persistent cache
    pub fn new(name: &str, device_id: &str) -> Self {
        let channel = async_channel::unbounded();

        Self {
            tune_cache: TuneCache::new(name, device_id),
            logger: Logger::new(),
            channel,
            autotuning: HashSet::new(),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        self.tune_cache.fastest(key)
    }

    /// Fetch the fastest autotune operation index for an autotune key and validate the checksum.
    #[cfg(std_io)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) {
        if let AutotuneLogLevel::Full = self.logger.log_level_autotune() {
            self.logger
                .log_autotune(&format!("validate checksum key={key}, checksum={checksum}"));
        }
        self.tune_cache.validate_checksum(key, checksum)
    }

    /// Handle an autotune result message, see [`execute_autotune`]
    fn handle_result(&mut self, msg: AutotuneMessage<K>) {
        match msg {
            AutotuneMessage::Pending(key) => {
                self.tune_cache.mark_pending(key);
            }
            AutotuneMessage::Done {
                key,
                fastest_index,
                results,
                #[cfg(std_io)]
                checksum,
            } => {
                match self.logger.log_level_autotune() {
                    AutotuneLogLevel::Minimal => {
                        let top_times = results
                            .iter()
                            .map(|r| {
                                let time = r
                                    .as_ref()
                                    .map(|r| r.computation.median)
                                    .unwrap_or(Duration::MAX);

                                let index = r.as_ref().map(|r| r.index).unwrap_or_default();
                                (index, time)
                            })
                            .take(3)
                            .collect::<Vec<_>>();

                        let result = results
                            .first()
                            .expect("At least one kernel needed.")
                            .as_ref()
                            .expect("At least one kernel has to succeed.");

                        self.logger.log_autotune(&format!(
                            "Fastest result {}-{key}. \n Top 3 times: {top_times:?}",
                            result.name,
                        ));
                    }
                    AutotuneLogLevel::Full => {
                        let result = results
                            .first()
                            .expect("At least one kernel needed.")
                            .as_ref()
                            .expect("At least one kernel has to succeed.");

                        self.logger
                            .log_autotune(&format!("Fastest result {}-{key}.", result.name,));

                        for result in results.iter() {
                            match result {
                                Ok(val) => {
                                    self.logger.log_autotune(&format!("{val}"));
                                }
                                Err(err) => self.logger.log_autotune(&format!("{err:?}")),
                            }
                        }
                    }
                    AutotuneLogLevel::Disabled => {}
                };

                self.tune_cache.cache_insert(key.clone(), fastest_index);

                #[cfg(std_io)]
                {
                    self.tune_cache
                        .persistent_cache_insert(key, checksum, fastest_index, results);
                }
            }
        }
    }

    /// Check if any autotuning results have come in asynchronously.
    pub fn handle_results(&mut self) {
        // Handle any results that have come in. Note that execute_autotune pushes results to the channel immediately if possible.
        // Since this function takes an &mut we know we have exclusive access, and no other threads are currently still adding results.
        while let Ok(msg) = self.channel.1.try_recv() {
            self.handle_result(msg);
        }
    }

    /// Execute benchmarks to find out what the fastest operation is.
    pub fn execute_autotune<
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
        In: Clone + Send + 'static,
        Out: AutotuneOutput,
    >(
        &self,
        key: K,
        inputs: &In,
        tunables: &TunableSet<K, In, Out>,
        client: &ComputeClient<S, C>,
    ) {
        log::info!("Tuning {key}");

        let autotunables = tunables.autotunables();
        let mut results = Vec::with_capacity(autotunables.len());

        for _ in 0..autotunables.len() {
            results.push(Err(AutotuneError::Skip));
        }

        let plan = tunables.plan(&key);
        let client = client.clone();

        let message = 'message: {
            if autotunables.len() == 1 {
                break 'message AutotuneMessage::Done {
                    key,
                    fastest_index: 0,
                    results,
                    #[cfg(std_io)]
                    checksum: tunables.compute_checksum(),
                };
            }

            let test_inputs = tunables.generate_inputs(&key, inputs);
            #[cfg(std_io)]
            let checksum = tunables.compute_checksum();

            let fut_result = async move {
                Self::generate_tune_message(
                    key,
                    &client,
                    plan,
                    autotunables,
                    test_inputs,
                    results,
                    #[cfg(std_io)]
                    checksum,
                )
                .await
            };

            cfg_if::cfg_if! {
                if #[cfg(target_family = "wasm")] {
                    let sender = self.channel.0.clone();
                    let send_fut = async move {
                        // If the channel has been closed, ignore. Maybe the main app is exiting
                        // before the tune results come in.
                        let _ = sender.send(fut_result.await).await;
                    };
                    // On wasm, spawn the tuning as a detached task.
                    wasm_bindgen_futures::spawn_local(send_fut);
                    // Mark the current tuning as pending.
                    AutotuneMessage::Pending(key)
                } else {
                    // On native, it is possible to run the tuning on a thread, which could help startup times,
                    // but it could be strange, as benchmarks would need a "warmup" time until a good kernel is selected.
                    cubecl_common::future::block_on(fut_result)
                }
            }
        };

        // Note that this message will be processed straight away by handle_results.
        self.channel
            .0
            .try_send(message)
            .expect("Loss message channel somehow");
    }

    async fn generate_tune_message<
        In: Clone + Send + 'static,
        Out: AutotuneOutput,
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    >(
        key: K,
        client: &ComputeClient<S, C>,
        mut plan: TunePlan,
        autotunables: Vec<Arc<dyn TuneFn<Inputs = In, Output = Out> + 'static>>,
        test_inputs: In,
        mut results: Vec<Result<AutotuneOutcome, AutotuneError>>,
        #[cfg(std_io)] checksum: String,
    ) -> AutotuneMessage<K> {
        Self::execute_tune_plan(client, &mut plan, autotunables, &test_inputs, &mut results).await;

        // Finds the fastest operation (by the median time).
        results.sort_by(|a, b| {
            let a = a
                .as_ref()
                .map(|r| r.computation.median)
                .unwrap_or(Duration::MAX);
            let b = b
                .as_ref()
                .map(|r| r.computation.median)
                .unwrap_or(Duration::MAX);

            a.cmp(&b)
        });

        // Log & send results.
        let result = results
            .first()
            .expect("At least one kernel needed.")
            .as_ref()
            .expect("At least one kernel has to succeed.");

        AutotuneMessage::Done {
            key,
            fastest_index: result.index,
            results,
            #[cfg(std_io)]
            checksum,
        }
    }

    async fn execute_tune_plan<
        In: Clone + Send + 'static,
        Out: AutotuneOutput,
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    >(
        client: &ComputeClient<S, C>,
        plan: &mut TunePlan,
        autotunables: Vec<Arc<dyn TuneFn<Inputs = In, Output = Out> + 'static>>,
        test_inputs: &In,
        results: &mut [Result<AutotuneOutcome, AutotuneError>],
    ) {
        loop {
            let mut num_autotuned = 0;

            let tunable_indices = plan.next();

            if tunable_indices.is_empty() {
                panic!("No autotune was flagged as valid for the problem.")
            }

            for index in tunable_indices {
                let op = &autotunables[index];
                let name = op.name().to_string();
                let tuner = TuneBenchmark::new(op.clone(), test_inputs.clone(), client.clone());
                let profiles = tuner.profile().map(|bench| (name, index, bench));

                match profiles {
                    Ok(result) => {
                        // Wait for the results to come in, and determine the outcome.
                        let (name, index, profiles) = result;
                        let result = Self::process_autotune(name, index, profiles).await;
                        match result {
                            Ok(val) => {
                                results[index] = Ok(val);
                                num_autotuned += 1;
                            }
                            Err(err) => {
                                results[index] = Err(err);
                            }
                        }
                    }
                    Err(err) => {
                        results[index] = Err(err);
                    }
                }
            }

            if num_autotuned > 0 {
                break;
            }
        }
    }

    async fn process_autotune(
        name: String,
        index: usize,
        profiles: Vec<ProfileDuration>,
    ) -> Result<AutotuneOutcome, AutotuneError> {
        let mut durations = Vec::new();
        if !profiles.is_empty() {
            let timing_method = profiles.first().unwrap().timing_method();
            for profile in profiles {
                durations.push(profile.resolve().await.duration());
            }
            let bench_durations = BenchmarkDurations::from_durations(timing_method, durations);

            Ok(AutotuneOutcome::new(
                name,
                index,
                BenchmarkComputations::new(&bench_durations),
            ))
        } else {
            Err(AutotuneError::Unknown(format!(
                "Runtime error while profiling {name}."
            )))
        }
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
