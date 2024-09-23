#[cfg(target_family = "wasm")]
use web_time::Duration;

#[cfg(not(target_family = "wasm"))]
use core::time::Duration;
use core::{any::Any, mem::ManuallyDrop};
#[cfg(feature = "std")]
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use cubecl_common::benchmark::{Benchmark, BenchmarkComputations, BenchmarkDurations};

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
}

#[allow(clippy::new_without_default)]
impl<K: AutotuneKey> Tuner<K> {
    /// Returns a tuner with cache initialized from persistent cache
    pub fn new(name: &str, device_id: &str) -> Self {
        Self {
            tune_cache: TuneCache::new(name, device_id),
        }
    }

    /// Fetch the fastest autotune operation index for an autotune key.
    pub fn autotune_fastest(&self, key: &K) -> Option<usize> {
        self.tune_cache.find_fastest(key)
    }

    /// Execute the fastest autotune operation if known, otherwise perform some benchmarks before.
    pub fn execute_autotune<S, C, Out>(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K, Out>>,
        client: &ComputeClient<S, C>,
    ) -> Out
    where
        S: ComputeServer,
        C: ComputeChannel<S>,
    {
        let operation = match self.tune_cache.try_cache(autotune_operation_set) {
            super::TuneCacheResult::Hit(ops) => ops,
            super::TuneCacheResult::Miss(set) => self.autotuning(set, client),
        };

        AutotuneOperation::<Out>::execute(operation)
    }

    fn autotuning<S, C, Out>(
        &mut self,
        autotune_operation_set: Box<dyn AutotuneOperationSet<K, Out>>,
        client: &ComputeClient<S, C>,
    ) -> Box<dyn AutotuneOperation<Out>>
    where
        S: ComputeServer,
        C: ComputeChannel<S>,
    {
        let key = autotune_operation_set.key();
        let autotunables = autotune_operation_set.autotunables();
        let mut names = Vec::with_capacity(autotunables.len());

        let results: Vec<Result<BenchmarkDurations, BenchError>> = autotunables
            .into_iter()
            .enumerate()
            .map(|(i, op)| {
                names.push(op.name().to_string());
                if autotune_operation_set.should_run(&key, i) {
                    self.run_benchmark(op, client)
                } else {
                    Ok(BenchmarkDurations::new(Vec::from([Duration::MAX])))
                }
            })
            .collect();

        #[cfg(feature = "std")]
        if results.iter().all(|it| it.is_err()) {
            let first_error = results.into_iter().next().unwrap().err().unwrap();
            resume_unwind(ManuallyDrop::into_inner(first_error));
        }

        // Finds the fastest operation, stores it and returns it
        let fastest_index = self.find_fastest(results);
        let fastest_name = names.get(fastest_index).unwrap();
        log::info!("Fastest result {fastest_name}-{key}");

        self.tune_cache.cache_insert(key.clone(), fastest_index);
        #[cfg(autotune_persistent_cache)]
        {
            let checksum = autotune_operation_set.compute_checksum();
            self.tune_cache
                .persistent_cache_insert(key, checksum, fastest_index);
            self.tune_cache.save();
        }

        match self.tune_cache.try_cache(autotune_operation_set) {
            super::TuneCacheResult::Hit(ops) => ops,
            super::TuneCacheResult::Miss(_) => panic!("We just inserted, should not miss"),
        }
    }

    fn run_benchmark<S, C, Out>(
        &mut self,
        operation: Box<dyn AutotuneOperation<Out>>,
        client: &ComputeClient<S, C>,
    ) -> Result<BenchmarkDurations, BenchError>
    where
        S: ComputeServer,
        C: ComputeChannel<S>,
    {
        #[cfg(feature = "std")]
        {
            catch_unwind(AssertUnwindSafe(|| {
                TuneBenchmark::new(operation, client.clone()).run()
            }))
            .map_err(|e| {
                println!("Caught error while benchmarking, falling back to next operation.");
                ManuallyDrop::new(e)
            })
        }
        #[cfg(not(feature = "std"))]
        Ok(TuneBenchmark::new(operation, client.clone()).run())
    }

    fn find_fastest(&self, results: Vec<Result<BenchmarkDurations, BenchError>>) -> usize {
        let mut smallest_duration = Duration::MAX;
        let mut fastest_tunable = None;

        for (i, result) in results.into_iter().enumerate() {
            let result = match result {
                Ok(result) => result,
                Err(_) => continue,
            };
            let computed = BenchmarkComputations::new(&result);

            if computed.median < smallest_duration {
                smallest_duration = computed.median;
                fastest_tunable = Some(i);
            }
        }

        fastest_tunable.expect("At least one kernel needed. ")
    }
}
