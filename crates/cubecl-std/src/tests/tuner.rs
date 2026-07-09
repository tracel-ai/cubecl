#[macro_export]
macro_rules! testgen_tuner {
    () => {
        mod tuner {
            use super::*;
            use alloc::sync::Arc;
            use core::sync::atomic::{AtomicUsize, Ordering};
            use cubecl_core::ir::{ElemType, FloatKind};
            use cubecl_core::prelude::*;
            use cubecl_runtime::throughput::{ThroughputKey, ThroughputMode};
            use cubecl_runtime::tune::{
                AutotuneBound, CloneInputGenerator, Tunable, TunableSet, TuneCacheResult, Tuner,
            };
            use cubecl_std::throughput::measure_peak_throughput;
            use cubecl_std::throughput::memory_direct::memory_direct_throughput;

            type Inputs = (
                cubecl_runtime::server::Handle,
                cubecl_runtime::server::Handle,
            );

            /// How many times each candidate streams the whole buffer. Kept small: the
            /// buffer is sized like the peak-throughput benchmark's, so one pass already
            /// moves hundreds of megabytes.
            const ITERATIONS: usize = 2;

            /// Fraction of the measured peak a candidate must reach to short-circuit the
            /// plan. Loose enough to absorb the gap between a peak (min-of-many) sample
            /// and an autotune (median-of-ten) sample.
            const THRESHOLD: f32 = 1.0;

            /// The tuner short-circuits its plan as soon as a candidate reaches the
            /// throughput bound, leaving later candidates unbenchmarked.
            #[$crate::tests::test_log::test]
            fn test_tuner_short_circuit_with_memory_direct() {
                let client = TestRuntime::client(&Default::default());
                let dtype = ElemType::Float(FloatKind::F32);

                // Mirror `memory_direct::build_kernel`'s geometry. The bound below is
                // stated relative to the peak that same kernel measures, so anything but
                // an identical buffer size and launch config makes the two incomparable.
                let hardware = &client.properties().hardware;
                let plane_size = hardware.plane_size_max.max(1);
                let cube_dim = (hardware.max_units_per_cube / plane_size * plane_size)
                    .max(plane_size)
                    .min(hardware.max_cube_dim.0) as usize;
                let cube_count = (hardware.num_streaming_multiprocessors.unwrap_or(64) * 32)
                    .min(hardware.max_cube_count.0) as usize;
                let vector_size = client
                    .io_optimized_vector_sizes(dtype.size())
                    .next()
                    .unwrap_or(1);

                const TARGET_BYTES: usize = 256 * 1024 * 1024;
                let line_bytes = vector_size * dtype.size();
                let target = TARGET_BYTES.min(client.properties().memory.max_page_size as usize);

                // `.max(total_threads)` is load-bearing: the kernel computes
                // `len - ABSOLUTE_POS`, which underflows for every thread that has no line
                // to copy, sending it into a long bounds-checked loop that moves no memory.
                let total_threads = cube_count * cube_dim;
                let num_lines = (target / line_bytes).max(total_threads);
                let bytes = num_lines * line_bytes;

                let in_handle = client.empty(bytes);
                let out_handle = client.empty(bytes);

                // All three candidates copy the same buffer the same number of times, and
                // differ only in how many cubes they spread that work across — the way real
                // autotune candidates differ. `slow` starves the device with a single cube.
                let make_op = |name: &'static str, cubes: usize, counter: Arc<AtomicUsize>| {
                    let client = client.clone();
                    Tunable::<String, Inputs, ()>::new(name, move |inputs: Inputs| {
                        counter.fetch_add(1, Ordering::Relaxed);
                        unsafe {
                            memory_direct_throughput::launch_unchecked::<TestRuntime>(
                                &client,
                                CubeCount::Static(cubes as u32, 1, 1),
                                CubeDim::new(&client, cube_dim),
                                vector_size,
                                BufferArg::from_raw_parts(inputs.0.clone(), num_lines),
                                BufferArg::from_raw_parts(inputs.1.clone(), num_lines),
                                ITERATIONS,
                                dtype.into(),
                            )
                        }
                        Ok::<(), String>(())
                    })
                };

                let execs_slow = Arc::new(AtomicUsize::new(0));
                let execs_fast = Arc::new(AtomicUsize::new(0));
                let execs_never = Arc::new(AtomicUsize::new(0));

                let peak = measure_peak_throughput(
                    &client,
                    ThroughputKey {
                        mode: ThroughputMode::Memory,
                        dtype,
                    },
                );

                // Same unit as `ThroughputValue::ops_per_s`: element reads plus writes.
                let bounds = vec![AutotuneBound {
                    ops_count: ITERATIONS * 2 * num_lines * vector_size,
                    throughput: peak.ops_per_s(),
                    threshold: THRESHOLD,
                }];

                let test_set = TunableSet::new(
                    |_inputs: &Inputs| "memory_direct_test".to_string(),
                    CloneInputGenerator,
                )
                .with(make_op("slow", 1, execs_slow.clone()))
                .with(make_op("fast", cube_count, execs_fast.clone()))
                .with(make_op("never", cube_count, execs_never.clone()));
                // .with_bounds(bounds);

                // A fresh cache namespace per run, otherwise the persistent autotune cache
                // turns every run after the first into a hit and nothing gets benchmarked.
                let nonce = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                let tuner = Tuner::<String>::new(&format!("test_tuner_{nonce}"), "test_device");

                let key = "memory_direct_test".to_string();
                let inputs = (in_handle, out_handle);
                let result =
                    tuner.check_tune(&key, &inputs, &test_set, || "checksum".to_string(), &client);

                assert!(
                    execs_slow.load(Ordering::Relaxed) > 0,
                    "The single-cube candidate should have been benchmarked"
                );
                assert!(
                    execs_fast.load(Ordering::Relaxed) > 0,
                    "The full-occupancy candidate should have been benchmarked"
                );
                assert_eq!(
                    execs_never.load(Ordering::Relaxed),
                    0,
                    "Reaching the throughput bound on `fast` should have short-circuited \
                     the plan before `never` was benchmarked"
                );

                match result {
                    TuneCacheResult::Hit { fastest_index } => assert_eq!(
                        fastest_index, 1,
                        "`fast` should have been selected, not index {fastest_index}"
                    ),
                    other => panic!("The tuner should have committed a result, got {other:?}"),
                }
            }
        }
    };
}
