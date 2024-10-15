use async_channel::{Receiver, Sender};

use core::future::Future;
use core::{any::Any, mem::ManuallyDrop};
use web_time::Duration;

#[cfg(all(not(target_family = "wasm"), feature = "std"))]
use std::panic::{resume_unwind, AssertUnwindSafe};

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use cubecl_common::benchmark::{BenchmarkComputations, BenchmarkDurations};

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
        let set = match self.tune_cache.try_cache(set) {
            // Cache hit -> return straight away.
            super::TuneCacheResult::Hit(ops) => return AutotuneOperation::execute(ops),
            // Pending -> wait for the cache to be filled.
            super::TuneCacheResult::Pending(set) => set,
            // Never seen before -> Start autotuning.
            super::TuneCacheResult::Miss(set) => {
                self.start_autotuning(set.as_ref(), client);
                set
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
        let operation = match self.tune_cache.try_cache(set) {
            super::TuneCacheResult::Hit(ops) => ops,
            super::TuneCacheResult::Pending(set) => set.fastest(0),
            super::TuneCacheResult::Miss(set) => set.fastest(0),
        };

        AutotuneOperation::execute(operation)
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
        self.tune_cache.mark_pending(key.clone());

        // let set = set.clone();
        let client = client.clone();
        let sender = self.results_send.clone();

        spawn_benchmark_task(async move {
            let autotunables = set.autotunables();
            let names: Vec<_> = autotunables
                .iter()
                .map(|op| op.name().to_string())
                .collect();

            let mut bench_times = vec![];
            for (i, op) in autotunables.into_iter().enumerate() {
                let res = Self::run_benchmark(set, &key, i, op, &client).await;
                let timings = res.map(|t| (i, BenchmarkComputations::new(&t), t));
                bench_times.push(timings);
            }

            // Panic if all tuners panicked.
            #[cfg(all(feature = "std", not(target_family = "wasm")))]
            if bench_times.iter().all(|it| it.is_err()) {
                let first_error = bench_times.into_iter().next().unwrap().err().unwrap();
                resume_unwind(ManuallyDrop::into_inner(first_error));
            }

            // Finds the fastest operation (by the median time).
            let mut bench_times: Vec<_> = bench_times.into_iter().filter_map(|b| b.ok()).collect();
            bench_times.sort_by_key(|p| p.1.median);

            // Log & send results.
            let fastest_index = bench_times.first().expect("At least one kernel needed. ").0;
            let fastest_name = &names[fastest_index];
            let top_times = bench_times.iter().map(|x| &x.1).take(3).collect::<Vec<_>>();
            log::info!("Fastest result {fastest_name}-{key}. \n Top 3 times: {top_times:?}",);

            sender
                .try_send(AutotuneMessage { key, fastest_index })
                .expect("Autotune results channel closed");
        });
    }

    async fn run_benchmark<S: ComputeServer, C: ComputeChannel<S>, Out>(
        set: &dyn AutotuneOperationSet<K, Out>,
        key: &K,
        index: usize,
        operation: Box<dyn AutotuneOperation<Out>>,
        client: &ComputeClient<S, C>,
    ) -> Result<BenchmarkDurations, BenchError> {
        if set.should_run(key, index) {
            let tuner = TuneBenchmark::new(operation, client.clone());

            #[cfg(any(target_family = "wasm", not(feature = "std")))]
            {
                Ok(tuner.sample_durations().await)
            }

            #[cfg(all(not(target_family = "wasm"), feature = "std"))]
            {
                use futures_lite::FutureExt;

                AssertUnwindSafe(tuner.sample_durations())
                    .catch_unwind()
                    .await
                    .map_err(|e| {
                        log::warn!(
                            "Caught error while benchmarking, falling back to next operation."
                        );
                        ManuallyDrop::new(e)
                    })
            }
        } else {
            Ok(BenchmarkDurations::new(vec![Duration::MAX]))
        }
    }
}

fn spawn_benchmark_task(future: impl Future<Output = ()>) {
    // Spawn the tuning as a task.
    #[cfg(target_family = "wasm")]
    wasm_bindgen_futures::spawn_local(future);

    // It is totally possible here to run the tuning on a thread, which woulp startup time,
    // but might have two downsides:
    // - Benchmarks now need a "warmup" time until a good kernel is selected.
    // - Tuning might be less precise, as it's possible that other operations are
    //   submitted while tuning, which might skew results.
    #[cfg(not(target_family = "wasm"))]
    futures_lite::future::block_on(future);
}
