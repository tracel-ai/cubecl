use async_channel::Sender;
use cubecl_common::future;

use core::future::Future;
use core::marker::PhantomData;
use core::{any::Any, mem::ManuallyDrop};
use cubecl_common::stub::Duration;

#[cfg(all(not(target_family = "wasm"), feature = "std"))]
use std::panic::resume_unwind;

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations, TimingMethod};

use crate::channel::ComputeChannel;
use crate::client::ComputeClient;
use crate::server::ComputeServer;
use crate::tune::{AutotuneOperation, AutotuneOperationSet, TuneBenchmark, TuneCache};

use super::{AutotuneKey, TuneCacheResult};

/// An error that occurred during benchmarking. If other benches succeeded, ignore this bench and
/// continue gracefully. If all benches fail, panic.
/// This error cannot be acted on in any way, because it's an opaque unwind object, and must be
/// `ManuallyDrop` because dropping it can cause unwinding to proceed. It can only
/// be passed to `resume_unwind` to continue the panic.
type BenchError = ManuallyDrop<Box<dyn Any + Send>>;

#[derive(Debug)]
/// Executes autotune benchmarking and caching
pub struct Tuner<K: AutotuneKey> {
    tune_cache: TuneCache<K>,
    #[cfg(target_family = "wasm")]
    wasm_restuls: wasm_fix::WasmDeferResults<K>,
}

#[cfg(target_family = "wasm")]
mod wasm_fix {
    use super::*;

    #[derive(Debug)]
    pub struct WasmDeferResults<K> {
        pub(crate) sender: async_channel::Sender<AutotuneMessage<K>>,
        pub(crate) receiver: async_channel::Receiver<AutotuneMessage<K>>,
    }

    impl<K> WasmDeferResults<K> {
        pub fn new() -> Self {
            let (sender, receiver) = async_channel::bounded(1);

            Self { sender, receiver }
        }
    }
}

struct AutotuneMessage<K> {
    key: K,
    fastest_index: usize,
}

/// Result from running benchmarks.
pub struct AutotuneResponse<K, Out> {
    key: K,
    fastest_index: usize,
    set: Box<dyn AutotuneOperationSet<K, Out>>,
}

/// Error from running autotune.
pub enum AutotuneError<K, Out> {
    /// The autotune tasks is spawn asynchonously, but can't be block to know the result.
    ///
    /// Should fall back on the default operation.
    #[cfg(target_family = "wasm")]
    Deferred(Box<dyn AutotuneOperationSet<K, Out>>),
    /// An unknown error happended.
    Unknown(String, (PhantomData<K>, PhantomData<Out>)),
}

/// Result returned from running autotune.
pub type AutotuneResult<K, Out> = Result<AutotuneResponse<K, Out>, AutotuneError<K, Out>>;

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Returns a tuner with cache initialized from persistent cache
    pub fn new(name: &str, device_id: &str) -> Self {
        Self {
            tune_cache: TuneCache::new(name, device_id),
            #[cfg(target_family = "wasm")]
            wasm_restuls: wasm_fix::WasmDeferResults::new(),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn fastest(&self, key: &K) -> TuneCacheResult {
        self.tune_cache.fastest(key)
    }

    /// Resolve async autotune launches.
    #[cfg(target_family = "wasm")]
    pub fn resolve(&mut self, key: &K) -> usize {
        let mut output = 0;

        while let Ok(msg) = self.wasm_restuls.receiver.try_recv() {
            if key == &msg.key {
                output = msg.fastest_index;
            }
            self.tune_cache
                .cache_insert(msg.key.clone(), msg.fastest_index);
        }

        output
    }

    /// Registers the [results](AutotuneResult) from [execute_autotune()](Self::execute_autotune).
    pub fn register_autotune<Out: Send>(&mut self, result: AutotuneResponse<K, Out>) -> Out {
        self.tune_cache
            .cache_insert(result.key.clone(), result.fastest_index);

        #[cfg(autotune_persistent_cache)]
        {
            let checksum = result.set.compute_checksum();
            self.tune_cache
                .persistent_cache_insert(result.key, checksum, result.fastest_index);
            self.tune_cache.save();
        }
        let op = result.set.fastest(result.fastest_index);

        AutotuneOperation::execute(op)
    }

    #[cfg(autotune_persistent_cache)]
    /// Fetch the fastest autotune operation index for an autotune key and validate the checksum.
    pub fn fastest_with_checksum<Out: Send>(
        &mut self,
        set: &dyn AutotuneOperationSet<K, Out>,
    ) -> TuneCacheResult {
        self.tune_cache.fastest_with_checksum(set)
    }

    /// Execute the fastest autotune operation if known, otherwise perform some benchmarks before.
    pub fn execute_autotune<S, C, Out: Send + 'static>(
        &self,
        set: Box<dyn AutotuneOperationSet<K, Out>>,
        client: &ComputeClient<S, C>,
    ) -> AutotuneResult<K, Out>
    where
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    {
        #[cfg(not(target_family = "wasm"))]
        {
            let (send, rec) = async_channel::bounded(1);

            self.start_autotuning(send, set.as_ref(), client);

            match rec.try_recv() {
                Ok(msg) => Ok(AutotuneResponse {
                    key: msg.key,
                    fastest_index: msg.fastest_index,
                    set,
                }),
                Err(err) => Err(AutotuneError::Unknown(
                    format!("Failed to run autotune {err:?}"),
                    Default::default(),
                )),
            }
        }

        #[cfg(target_family = "wasm")]
        {
            self.start_autotuning(self.wasm_restuls.sender.clone(), set.as_ref(), client);
            Err(AutotuneError::Deferred(set))
        }
    }

    fn start_autotuning<
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
        Out: Send + 'static,
    >(
        &self,
        sender: Sender<AutotuneMessage<K>>,
        set: &dyn AutotuneOperationSet<K, Out>,
        client: &ComputeClient<S, C>,
    ) {
        let key = set.key();
        log::info!("Tuning {key}");

        let autotunables: Vec<_> = set
            .autotunables()
            .into_iter()
            .enumerate()
            .map(|(index, op)| (index, op, set.should_run(&key, index)))
            .collect();
        let client = client.clone();

        spawn_benchmark_task(async move {
            #[derive(new, Debug)]
            struct BenchResult {
                name: String,
                index: usize,
                computation: BenchmarkComputations,
            }

            let mut bench_results = Vec::with_capacity(autotunables.len());

            for (index, op, should_run) in autotunables.into_iter() {
                let name = op.name().to_string();
                let result = Self::run_benchmark(op, &client, should_run)
                    .await
                    .map(|durations| {
                        log::info!("Name: {name} => {}", durations);
                        BenchResult::new(name, index, BenchmarkComputations::new(&durations))
                    });

                bench_results.push(result);
            }

            // Panic if all tuners panicked.
            #[cfg(all(feature = "std", not(target_family = "wasm")))]
            if bench_results.iter().all(|result| result.is_err()) {
                let first_error = bench_results.into_iter().next().unwrap().err().unwrap();
                resume_unwind(ManuallyDrop::into_inner(first_error));
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
                .try_send(AutotuneMessage { key, fastest_index })
                .expect("Autotune results channel closed");
        });
    }

    async fn run_benchmark<S: ComputeServer, C: ComputeChannel<S>, Out>(
        operation: Box<dyn AutotuneOperation<Out>>,
        client: &ComputeClient<S, C>,
        should_run: bool,
    ) -> Result<BenchmarkDurations, BenchError> {
        if should_run {
            let tuner = TuneBenchmark::new(operation, client.clone());
            future::catch_unwind(tuner.sample_durations())
                .await
                .map_err(|e| {
                    log::warn!("Caught error while benchmarking, falling back to next operation.");
                    ManuallyDrop::new(e)
                })
        } else {
            Ok(BenchmarkDurations::new(
                TimingMethod::DeviceOnly,
                vec![Duration::MAX],
            ))
        }
    }
}

fn spawn_benchmark_task(future: impl Future<Output = ()> + 'static) {
    // Spawn the tuning as a task.
    #[cfg(target_family = "wasm")]
    {
        wasm_bindgen_futures::spawn_local(future);
    }

    // It is totally possible here to run the tuning on a thread, which woulp startup time,
    // but might have two downsides:
    // - Benchmarks now need a "warmup" time until a good kernel is selected.
    // - Tuning might be less precise, as it's possible that other operations are
    //   submitted while tuning, which might skew results.
    #[cfg(not(target_family = "wasm"))]
    future::block_on(future);
}
