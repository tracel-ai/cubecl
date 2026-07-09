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
                AutotuneBound, AutotuneKey, CloneInputGenerator, Tunable, TunableSet, Tuner,
            };
            use cubecl_std::throughput::measure_peak_throughput;
            use cubecl_std::throughput::memory_direct::memory_direct_throughput;

            #[$crate::tests::test_log::test]
            fn test_tuner_short_circuit_with_memory_direct() {
                const BUFFER_SIZE: usize = 1024 * 1024;

                let client = TestRuntime::client(&Default::default());

                let in_handle = client.empty(BUFFER_SIZE * core::mem::size_of::<f32>());
                let out_handle = client.empty(BUFFER_SIZE * core::mem::size_of::<f32>());

                let execs1 = Arc::new(AtomicUsize::new(0));
                let execs2 = Arc::new(AtomicUsize::new(0));
                let execs3 = Arc::new(AtomicUsize::new(0));

                let vector_size = client
                    .io_optimized_vector_sizes(core::mem::size_of::<f32>())
                    .next()
                    .unwrap_or(1);
                let vector_len = BUFFER_SIZE / (vector_size as usize);

                let hardware = &client.properties().hardware;

                let plane = hardware.plane_size_max.max(1);
                let cube_dim = (hardware.max_units_per_cube / plane * plane)
                    .max(plane)
                    .min(hardware.max_cube_dim.0);

                let cube_count = vector_len.div_ceil(cube_dim as usize).max(1);

                const ITERS_1: u32 = 10;
                const ITERS_2: u32 = 5;
                const ITERS_3: u32 = 1;

                let make_op = |name: &str, iters: u32, execs: Arc<AtomicUsize>| {
                    let client_clone = client.clone();
                    Tunable::<
                        String,
                        (
                            cubecl_runtime::server::Handle,
                            cubecl_runtime::server::Handle,
                        ),
                        (),
                    >::new(name, move |inputs| {
                        execs.fetch_add(1, Ordering::Relaxed);
                        unsafe {
                            memory_direct_throughput::launch_unchecked::<TestRuntime>(
                                &client_clone,
                                CubeCount::Static(cube_count as u32, 1, 1),
                                CubeDim::new(&client_clone, cube_dim as usize),
                                vector_size,
                                BufferArg::from_raw_parts(inputs.0.clone(), vector_len),
                                BufferArg::from_raw_parts(inputs.1.clone(), vector_len),
                                iters as usize,
                                ElemType::Float(FloatKind::F32).into(),
                            )
                        }
                        Ok::<(), String>(())
                    })
                };

                let op1 = make_op("slow", ITERS_1, execs1.clone());
                let op2 = make_op("fast", ITERS_2, execs2.clone());
                let op3 = make_op("never", ITERS_3, execs3.clone());

                let key = ThroughputKey {
                    mode: ThroughputMode::Memory,
                    dtype: ElemType::Float(FloatKind::F32),
                };
                let throughput_value = measure_peak_throughput(&client, key);

                let bounds = vec![AutotuneBound {
                    ops_count: 2 * BUFFER_SIZE * (ITERS_2 as usize),
                    throughput: throughput_value.ops_per_s(),
                    threshold: 1.0,
                }];

                let test_set = TunableSet::new(
                    move |inputs: &(
                        cubecl_runtime::server::Handle,
                        cubecl_runtime::server::Handle,
                    )| { "memory_direct_test".to_string() },
                    CloneInputGenerator,
                )
                .with(op1)
                .with(op2)
                .with(op3)
                .with_bounds(bounds);

                let tuner = Tuner::<String>::new("test_tuner", "test_device");
                let inputs = (in_handle, out_handle);
                tuner.check_tune(
                    &"memory_direct_test".to_string(),
                    &inputs,
                    &test_set,
                    || "checksum".to_string(),
                    &client,
                );

                assert!(
                    execs1.load(Ordering::Relaxed) > 0,
                    "First kernel should have executed"
                );
                assert!(
                    execs2.load(Ordering::Relaxed) > 0,
                    "Second kernel should have executed"
                );
                assert_eq!(
                    execs3.load(Ordering::Relaxed),
                    0,
                    "Tuner should have short-circuited and only executed the first two kernels"
                );
            }
        }
    };
}
