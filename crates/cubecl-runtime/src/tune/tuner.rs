use alloc::boxed::Box;
use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;
use async_channel::{Receiver, Sender};
use cubecl_common::format::format_debug;
use cubecl_common::profile::ProfileDuration;
use hashbrown::HashSet;

use core::time::Duration;

use alloc::string::{String, ToString};
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};

use crate::config::{Logger, autotune::AutotuneLogLevel};
use crate::server::LaunchError;
use crate::tune::{AutotuneResult, TuneBenchmark, TuneCache};
use crate::{client::ComputeClient, runtime::Runtime};

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
        results: Vec<AutotuneResult>,
        #[cfg(std_io)]
        checksum: String,
        context_logs: Option<String>,
    },
    #[allow(dead_code)]
    Pending(K),
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
                context_logs,
            } => {
                match self.logger.log_level_autotune() {
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

                        let context = match &context_logs {
                            Some(context) => context,
                            None => "",
                        };
                        self.logger.log_autotune(&format!(
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

                        let context = match &context_logs {
                            Some(context) => context,
                            None => "",
                        };
                        self.logger.log_autotune(&format!(
                            "Fastest result {}-{key}. Context: {context}",
                            result.name,
                        ));

                        for result in results.iter() {
                            match &result.outcome {
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
    pub fn prepare_autotune<R: Runtime, In: Clone + Send + 'static, Out: AutotuneOutput>(
        &self,
        key: K,
        inputs: &In,
        tunables: &TunableSet<K, In, Out>,
        client: &ComputeClient<R>,
    ) -> Box<dyn FnOnce()> {
        log::info!("Tuning {key}");

        // Note that this message will be processed straight away by handle_results.
        let sender = self.channel.0.clone();

        let autotunables = tunables.autotunables();
        let mut results: Vec<AutotuneResult> = Vec::with_capacity(autotunables.len());

        for a in autotunables.iter() {
            results.push(AutotuneResult::error(AutotuneError::Skip {
                name: a.name().to_string(),
            }));
        }

        if autotunables.len() == 1 {
            let message = AutotuneMessage::Done {
                key,
                fastest_index: 0,
                results,
                #[cfg(std_io)]
                checksum: tunables.compute_checksum(),
                context_logs: None,
            };

            return Box::new(move || {
                sender
                    .try_send(message)
                    .expect("Loss message channel somehow")
            });
        }

        let client = client.clone();
        let key_cloned = key.clone();
        let plan = tunables.plan(&key);
        let inputs_generator = tunables.inputs_generator(&key.clone(), inputs);

        #[cfg(std_io)]
        let checksum = tunables.compute_checksum();
        let context_logs = match self.logger.log_level_autotune() {
            AutotuneLogLevel::Disabled => false,
            AutotuneLogLevel::Minimal => false,
            AutotuneLogLevel::Full => true,
        };

        let fut_result = async move {
            let test_inputs = inputs_generator();

            Self::generate_tune_message(
                key_cloned,
                &client,
                plan,
                autotunables,
                test_inputs,
                results,
                #[cfg(std_io)]
                checksum,
                context_logs,
            )
            .await
        };

        Box::new(move || {
            let message = {
                cfg_if::cfg_if! {
                    if #[cfg(target_family = "wasm")] {
                        let sender = sender.clone();

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
                        cubecl_common::future::block_on(fut_result)
                    }
                }
            };

            // Note that this message will be processed straight away by handle_results.
            sender
                .try_send(message)
                .expect("Loss message channel somehow");
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn generate_tune_message<In: Clone + Send + 'static, Out: AutotuneOutput, R: Runtime>(
        key: K,
        client: &ComputeClient<R>,
        mut plan: TunePlan,
        autotunables: Vec<Arc<dyn TuneFn<Inputs = In, Output = Out> + 'static>>,
        test_inputs: In,
        mut results: Vec<AutotuneResult>,
        #[cfg(std_io)] checksum: String,
        context_logs: bool,
    ) -> AutotuneMessage<K> {
        let context_logs = match Self::execute_tune_plan(
            client,
            &mut plan,
            autotunables,
            &test_inputs,
            &mut results,
            context_logs,
        )
        .await
        {
            Ok(context_logs) => context_logs,
            Err(err) => {
                panic!("Can't execute the autotune plan for key: {key:?}\n - Error: {err:?}");
            }
        };

        // Finds the fastest operation (by the median time).
        results.sort_by(|a, b| {
            let a = a
                .outcome
                .as_ref()
                .map(|r| r.computation.median)
                .unwrap_or(Duration::MAX);
            let b = b
                .outcome
                .as_ref()
                .map(|r| r.computation.median)
                .unwrap_or(Duration::MAX);

            a.cmp(&b)
        });

        // Log & send results.
        let result = results
            .first()
            .expect("At least one kernel needed.")
            .outcome
            .as_ref()
            .expect("At least one kernel has to succeed.");

        AutotuneMessage::Done {
            key,
            fastest_index: result.index,
            results,
            #[cfg(std_io)]
            checksum,
            context_logs,
        }
    }

    async fn execute_tune_plan<In: Clone + Send + 'static, Out: AutotuneOutput, R: Runtime>(
        client: &ComputeClient<R>,
        plan: &mut TunePlan,
        autotunables: Vec<Arc<dyn TuneFn<Inputs = In, Output = Out> + 'static>>,
        test_inputs: &In,
        results: &mut [AutotuneResult],
        context_logs: bool,
    ) -> Result<Option<String>, AutotuneError> {
        #[derive(Debug)]
        #[allow(unused_variables, dead_code)] // Only use for debug
        struct Context<'a> {
            plan: &'a TunePlan,
            results: &'a [AutotuneResult],
        }

        let mut context_logs = match context_logs {
            true => Some("".to_string()),
            false => None,
        };

        loop {
            let mut num_success = 0;
            let tunable_indices = plan.next(context_logs.as_mut());

            if tunable_indices.is_empty() {
                return Err(AutotuneError::NoValidKernelFound {
                    context: format_debug(&Context { plan, results }),
                });
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
                                results[index] = AutotuneResult::success(val);
                                num_success += 1;
                            }
                            Err(err) => {
                                results[index] = AutotuneResult::error(err);
                            }
                        }
                    }
                    Err(err) => {
                        results[index] = AutotuneResult::error(err);
                    }
                }
            }

            if num_success > 0 {
                break;
            }
        }

        Ok(context_logs)
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
            Err(AutotuneError::Unknown {
                name,
                err: "No profiling available".to_string(),
            })
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
