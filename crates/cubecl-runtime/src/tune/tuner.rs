use async_channel::{Receiver, Sender};

use core::{any::Any, mem::ManuallyDrop};
use web_time::Duration;

#[cfg(all(not(target_family = "wasm"), feature = "std"))]
use std::panic::{resume_unwind, AssertUnwindSafe};

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::sync::Arc;
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
    results_send: Sender<(K, usize)>,
    results_rec: Receiver<(K, usize)>,
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
        autotune_operation_set: Arc<dyn AutotuneOperationSet<K, Out>>,
        client: &ComputeClient<S, C>,
    ) -> Out
    where
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    {
        match self.tune_cache.try_cache(autotune_operation_set.clone()) {
            // Cache hit -> return straight away.
            super::TuneCacheResult::Hit(ops) => return AutotuneOperation::execute(ops),
            // Pending -> wait for the cache to be filled.
            super::TuneCacheResult::Pending => {}
            // Never seen before -> Start autotuning.
            super::TuneCacheResult::Miss => {
                self.start_autotuning(autotune_operation_set.clone(), client);
            }
        };

        // Collect any new cache results that have come in.
        while let Ok((key, fastest_index)) = self.results_rec.try_recv() {
            self.tune_cache.cache_insert(key.clone(), fastest_index);
            #[cfg(autotune_persistent_cache)]
            {
                let checksum = autotune_operation_set.compute_checksum();
                self.tune_cache
                    .persistent_cache_insert(key, checksum, fastest_index);
                self.tune_cache.save();
            }
        }

        // Check to see if value is now cached. If not, just use a default operation.
        let operation = match self.tune_cache.try_cache(autotune_operation_set.clone()) {
            super::TuneCacheResult::Hit(ops) => ops,
            _ => autotune_operation_set.fastest(0),
        };

        AutotuneOperation::execute(operation)
    }

    fn start_autotuning<S, C, Out: Send + 'static>(
        &mut self,
        autotune_operation_set: Arc<dyn AutotuneOperationSet<K, Out>>,
        client: &ComputeClient<S, C>,
    ) where
        S: ComputeServer + 'static,
        C: ComputeChannel<S> + 'static,
    {
        let key = autotune_operation_set.key();
        self.tune_cache.mark_pending(key.clone());

        let set = autotune_operation_set.clone();
        let client = client.clone();
        let sender = self.results_send.clone();

        let fut = async move {
            let fastest_index = {
                let autotunables = set.autotunables();
                let mut names = Vec::with_capacity(autotunables.len());

                let mut results: Vec<Result<BenchmarkDurations, BenchError>> = vec![];
                for (i, op) in autotunables.into_iter().enumerate() {
                    names.push(op.name().to_string());

                    let res = if set.should_run(&key, i) {
                        Self::run_benchmark(op, &client).await
                    } else {
                        Ok(BenchmarkDurations::new(Vec::from([Duration::MAX])))
                    };

                    results.push(res);
                }

                #[cfg(all(feature = "std", not(target_family = "wasm")))]
                if results.iter().all(|it| it.is_err()) {
                    let first_error = results.into_iter().next().unwrap().err().unwrap();
                    resume_unwind(ManuallyDrop::into_inner(first_error));
                }

                // Finds the fastest operation, stores it and returns it
                let mut bench_times: Vec<_> = results
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, result)| match result {
                        Ok(t) => Some((i, BenchmarkComputations::new(&t), t)),
                        Err(_) => None,
                    })
                    .collect();
                bench_times.sort_by_key(|p| p.1.median);

                let fastest_index = bench_times.first().expect("At least one kernel needed. ").0;
                let fastest_name = names.get(fastest_index).unwrap();
                log::info!(
                    "Fastest result {fastest_name}-{key}. \n Top 3 times: {:?}",
                    bench_times.iter().map(|x| &x.1).take(3).collect::<Vec<_>>()
                );
                fastest_index
            };

            sender
                .send((key, fastest_index))
                .await
                .expect("Autotune results channel closed");
        };

        // Spawn the tuning as a task.
        #[cfg(target_family = "wasm")]
        wasm_bindgen_futures::spawn_local(fut);

        // It is totally possible here to run the tuning on a thread, which woulp startup time,
        // but might have two downsides:
        // - Benchmarks now need a "warmup" time until a good kernel is selected.
        // - Tuning might be less precise, as it's possible that other operations are
        //   submitted while tuning, which might skew results.
        #[cfg(not(target_family = "wasm"))]
        futures_lite::future::block_on(fut);
    }

    async fn run_benchmark<S: ComputeServer, C: ComputeChannel<S>, Out>(
        operation: Box<dyn AutotuneOperation<Out>>,
        client: &ComputeClient<S, C>,
    ) -> Result<BenchmarkDurations, BenchError> {
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
                    println!("Caught error while benchmarking, falling back to next operation.");
                    ManuallyDrop::new(e)
                })
        }
    }
}
