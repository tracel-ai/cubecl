use async_channel::{Receiver, Sender};
use cubecl_common::future;

use core::future::Future;
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

use super::AutotuneKey;

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
    results_send: Sender<AutotuneMessage<K>>,
    results_rec: Receiver<AutotuneMessage<K>>,
}

struct AutotuneMessage<K> {
    key: K,
    fastest_index: usize,
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Returns a tuner with cache initialized from persistent cache
    pub fn new(name: &str, device_id: &str) -> Self {
        let (send, rec) = async_channel::unbounded();
        Self {
            results_send: send,
            results_rec: rec,
            tune_cache: TuneCache::new(name, device_id),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn autotune_fastest(&self, key: &K) -> Option<usize> {
        self.tune_cache.find_fastest(key)
    }

    /// Execute the fastest autotune operation if known, otherwise perform some benchmarks before.
    pub fn execute_autotune<S, C, Out: Send + 'static>(
        &mut self,
        set: Box<dyn AutotuneOperationSet<K, Out>>,
        client: &ComputeClient<S, C>,
    ) -> Out
    where
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    {
        match self.tune_cache.try_cache(set.as_ref()) {
            // Cache hit -> return straight away.
            super::TuneCacheResult::Hit { fastest_index } => {
                let op = set.fastest(fastest_index);
                return op.execute();
            }
            // Pending -> wait for the cache to be filled.
            super::TuneCacheResult::Pending => {}
            // Never seen before -> Start autotuning.
            super::TuneCacheResult::Miss => {
                self.start_autotuning(set.as_ref(), client);
            }
        };

        // Collect any new cache results that have come in.
        while let Ok(msg) = self.results_rec.try_recv() {
            self.tune_cache
                .cache_insert(msg.key.clone(), msg.fastest_index);

            #[cfg(autotune_persistent_cache)]
            {
                let checksum = set.compute_checksum();
                self.tune_cache
                    .persistent_cache_insert(msg.key, checksum, msg.fastest_index);
                self.tune_cache.save();
            }
        }

        // Check to see if value is now cached. If not, just use a default operation.
        let fastest_index = match self.tune_cache.try_cache(set.as_ref()) {
            super::TuneCacheResult::Hit { fastest_index } => fastest_index,
            _ => 0,
        };

        let op = set.fastest(fastest_index);
        AutotuneOperation::execute(op)
    }

    fn start_autotuning<
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
        Out: Send + 'static,
    >(
        &mut self,
        set: &dyn AutotuneOperationSet<K, Out>,
        client: &ComputeClient<S, C>,
    ) {
        let key = set.key();
        log::info!("Tuning {key}");
        self.tune_cache.mark_pending(key.clone());

        let autotunables: Vec<_> = set
            .autotunables()
            .into_iter()
            .enumerate()
            .map(|(index, op)| (index, op, set.should_run(&key, index)))
            .collect();
        let client = client.clone();
        let sender = self.results_send.clone();

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
