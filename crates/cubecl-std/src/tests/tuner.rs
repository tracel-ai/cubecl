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
                const BUFFER_SIZE: usize = 1024;

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

                let sms = hardware.num_streaming_multiprocessors.unwrap_or(64);
                let cube_count = (sms * 32).min(hardware.max_cube_count.0);

                let execs1_clone = execs1.clone();
                let client1 = client.clone();
                let op1 = Tunable::<
                    String,
                    (
                        cubecl_runtime::server::Handle,
                        cubecl_runtime::server::Handle,
                    ),
                    (),
                >::new("slow", move |inputs| {
                    execs1_clone.fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        memory_direct_throughput::launch_unchecked::<TestRuntime>(
                            &client1,
                            CubeCount::Static(cube_count as u32, 1, 1),
                            CubeDim::new(&client1, cube_dim as usize),
                            vector_size,
                            BufferArg::from_raw_parts(inputs.0.clone(), vector_len),
                            BufferArg::from_raw_parts(inputs.1.clone(), vector_len),
                            100000, // lots of iters
                            ElemType::Float(FloatKind::F32).into(),
                        )
                    }
                    Ok::<(), String>(())
                });

                let execs2_clone = execs2.clone();
                let client2 = client.clone();
                let op2 = Tunable::<
                    String,
                    (
                        cubecl_runtime::server::Handle,
                        cubecl_runtime::server::Handle,
                    ),
                    (),
                >::new("fast", move |inputs| {
                    execs2_clone.fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        memory_direct_throughput::launch_unchecked::<TestRuntime>(
                            &client2,
                            CubeCount::Static(cube_count as u32, 1, 1),
                            CubeDim::new(&client2, cube_dim as usize),
                            vector_size,
                            BufferArg::from_raw_parts(inputs.0.clone(), vector_len),
                            BufferArg::from_raw_parts(inputs.1.clone(), vector_len),
                            1000, // 1000 iter
                            ElemType::Float(FloatKind::F32).into(),
                        )
                    }
                    Ok::<(), String>(())
                });

                let execs3_clone = execs3.clone();
                let client3 = client.clone();
                let op3 = Tunable::<
                    String,
                    (
                        cubecl_runtime::server::Handle,
                        cubecl_runtime::server::Handle,
                    ),
                    (),
                >::new("never", move |inputs| {
                    execs3_clone.fetch_add(1, Ordering::Relaxed);
                    unsafe {
                        memory_direct_throughput::launch_unchecked::<TestRuntime>(
                            &client3,
                            CubeCount::Static(cube_count as u32, 1, 1),
                            CubeDim::new(&client3, cube_dim as usize),
                            vector_size,
                            BufferArg::from_raw_parts(inputs.0.clone(), vector_len),
                            BufferArg::from_raw_parts(inputs.1.clone(), vector_len),
                            1,
                            ElemType::Float(FloatKind::F32).into(),
                        )
                    }
                    Ok::<(), String>(())
                });

                let key = ThroughputKey {
                    mode: ThroughputMode::Memory,
                    dtype: ElemType::Float(FloatKind::F32),
                };
                let throughput_value = measure_peak_throughput(&client, key);

                let bounds = vec![AutotuneBound {
                    ops_count: 2 * BUFFER_SIZE * 1000,
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
