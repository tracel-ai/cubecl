use async_channel::{Receiver, Sender};
use cubecl_common::future;
use hashbrown::HashSet;

use core::future::Future;
use cubecl_common::stub::Duration;

#[cfg(not(target_family = "wasm"))]
use core::sync::atomic::{AtomicU64, Ordering};

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use cubecl_common::benchmark::BenchmarkComputations;

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use crate::tune::{TuneBenchmark, TuneCache};

use super::{AutotuneKey, TunableSet, TuneCacheResult};

#[derive(Debug)]
/// Executes autotune benchmarking and caching
pub struct Tuner<K: AutotuneKey> {
    tune_cache: TuneCache<K>,
    channel: (Sender<AutotuneMessage<K>>, Receiver<AutotuneMessage<K>>),
    pub(crate) autotuning: HashSet<K>,
    #[cfg(not(target_family = "wasm"))]
    current: AtomicU64,
}

#[cfg_attr(
    autotune_persistent_cache,
    derive(serde::Serialize, serde::Deserialize, PartialEq, Eq)
)]
#[derive(new, Debug, Clone)]
pub(crate) struct AutotuneOutcome {
    name: String,
    index: usize,
    computation: BenchmarkComputations,
}
/// Result from running benchmarks.
enum AutotuneMessage<K> {
    Done {
        key: K,
        fastest_index: usize,
        #[cfg(autotune_persistent_cache)]
        checksum: String,
        #[cfg(autotune_persistent_cache)]
        results: Vec<Result<AutotuneOutcome, String>>,
    },
    Starting {
        key: K,
    },
}

/// Error from running autotune.
#[derive(Debug)]
pub enum AutotuneError {
    /// An unknown error happened.
    Unknown(String),
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
            channel,
            autotuning: HashSet::new(),
            #[cfg(not(target_family = "wasm"))]
            current: 0.into(),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        self.tune_cache.fastest(key)
    }

    /// Fetch the fastest autotune operation index for an autotune key and validate the checksum.
    #[cfg(autotune_persistent_cache)]
    pub fn validate_checksum(&mut self, key: &K, checksum: &str) {
        self.tune_cache.validate_checksum(key, checksum)
    }

    /// Wait for async results to come in.
    pub fn resolve(&mut self) {
        #[cfg(not(target_family = "wasm"))]
        // On native platforms, we know exactly how many tasks to wait for.
        //
        // Those tasks can be registered from another thread, but since the tuner shares the same
        // state, we can wait for all results to be saved before deciding which kernel to launch.
        // This may happen if multiple threads trigger the same autotune task.
        while self.current.load(Ordering::Relaxed) > 0 {
            self.resolve_loop();
        }

        #[cfg(target_family = "wasm")]
        self.resolve_loop();
    }

    fn resolve_loop(&mut self) {
        while let Ok(msg) = self.channel.1.try_recv() {
            match msg {
                AutotuneMessage::Done {
                    key,
                    fastest_index,
                    #[cfg(autotune_persistent_cache)]
                    checksum,
                    #[cfg(autotune_persistent_cache)]
                    results,
                } => {
                    #[cfg(not(target_family = "wasm"))]
                    AtomicU64::fetch_sub(&self.current, 1, Ordering::Relaxed);

                    self.tune_cache.cache_insert(key.clone(), fastest_index);

                    #[cfg(autotune_persistent_cache)]
                    {
                        self.tune_cache.persistent_cache_insert(
                            key,
                            checksum,
                            fastest_index,
                            results,
                        );
                    }
                }
                AutotuneMessage::Starting { key } => {
                    self.tune_cache.mark_pending(key);
                }
            }
        }
    }

    /// Execute benchmarks to find out what the fastest operation is.
    pub fn execute_autotune<
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
        In: Clone + Send + 'static,
        Out: Send + 'static,
    >(
        &self,
        key: K,
        inputs: &In,
        tunables: &TunableSet<K, In, Out>,
        client: &ComputeClient<S, C>,
    ) {
        #[cfg(not(target_family = "wasm"))]
        AtomicU64::fetch_add(&self.current, 1, Ordering::Relaxed);

        log::info!("Tuning {key}");

        let autotunables: Vec<_> = tunables
            .autotunables()
            .iter()
            .cloned()
            .enumerate()
            .collect();

        let client = client.clone();
        let sender = self.channel.0.clone();

        if autotunables.len() == 1 {
            sender
                .try_send(AutotuneMessage::Done {
                    key,
                    fastest_index: autotunables[0].0,
                    #[cfg(autotune_persistent_cache)]
                    checksum: tunables.compute_checksum(),
                    #[cfg(autotune_persistent_cache)]
                    results: Vec::new(),
                })
                .expect("Autotune results channel closed");
            return;
        }

        sender
            .try_send(AutotuneMessage::Starting { key: key.clone() })
            .expect("Autotune results channel closed");

        #[cfg(autotune_persistent_cache)]
        let checksum = tunables.compute_checksum();

        let test_inputs = tunables.generate_inputs(&key, inputs);

        spawn_benchmark_task(async move {
            let mut bench_results = Vec::with_capacity(autotunables.len());

            for (index, op) in autotunables.into_iter() {
                let name = op.name().to_string();
                let tuner = TuneBenchmark::new(op, test_inputs.clone(), client.clone());

                let sample_fut = tuner.sample_durations();
                let result = sample_fut.await;
                let result = result.map(|durations| {
                    log::info!("Name: {name} => {}", durations);
                    AutotuneOutcome::new(name, index, BenchmarkComputations::new(&durations))
                });

                bench_results.push(result);
            }

            // Panic if all tuners panicked.
            #[cfg(all(feature = "std", not(target_family = "wasm")))]
            if bench_results.iter().all(|result| result.is_err()) {
                let first_error = bench_results.into_iter().next().unwrap().err().unwrap();

                match first_error {
                    AutotuneError::Unknown(reason) => panic!("{reason}"),
                }
            }

            // Finds the fastest operation (by the median time).
            bench_results.sort_by(|a, b| {
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
            let result = bench_results.first().expect("At least one kernel needed. ");

            let fastest_index = if let Ok(result) = result {
                let top_times = bench_results
                    .iter()
                    .map(|r| {
                        r.as_ref()
                            .map(|r| r.computation.median)
                            .unwrap_or(Duration::MAX)
                    })
                    .take(3)
                    .collect::<Vec<_>>();
                log::info!(
                    "Fastest result {}-{key}. \n Top 3 times: {top_times:?}",
                    result.name,
                );

                result.index
            } else {
                0
            };

            sender
                .send(AutotuneMessage::Done {
                    key,
                    fastest_index,
                    #[cfg(autotune_persistent_cache)]
                    checksum,
                    #[cfg(autotune_persistent_cache)]
                    results: bench_results
                        .into_iter()
                        .map(|result| result.map_err(|err| format!("{err:?}")))
                        .collect(),
                })
                .await
                .expect("Autotune results channel closed");
        });
    }
}

fn spawn_benchmark_task(future: impl Future<Output = ()> + Send + 'static) {
    // On wasm, spawn the tuning as a detached task.
    #[cfg(target_family = "wasm")]
    wasm_bindgen_futures::spawn_local(future);

    // On native, it is possible to run the tuning on a thread, which could help startup times,
    // but might have two downsides:
    // - Benchmarks would need a "warmup" time until a good kernel is selected.
    // - Tuning could be less precise, as it's possible that other operations are
    //   submitted while tuning, which might skew results.
    //
    // So, for now, just block on the future.
    #[cfg(not(target_family = "wasm"))]
    future::block_on(future);
}
