#[macro_use]
extern crate derive_new;

extern crate alloc;

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CpuRuntime;

    pub use half::f16;

    use cubecl_core as cubecl;
    use cubecl_core::prelude::*;
    use cubecl_environment::config::RuntimeConfig;
    use cubecl_environment::stream::StreamId;
    use cubecl_runtime::config::CubeClRuntimeConfig;

    cubecl_core::testgen_all!(f32: [f16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
    cubecl_std::testgen_tensor_into_contiguous!();
    cubecl_std::testgen_quantized_view!(f32);

    #[cube(launch)]
    fn barrier_smoke(out: &mut [f32]) {
        let barrier = barrier::Barrier::local();
        barrier.arrive_and_wait();
        if UNIT_POS == 0 {
            out[0] = 1.0;
        }
    }

    #[cube(launch)]
    fn sync_cube_magic(out: &mut [u32]) {
        let mut mem = Shared::new_slice(1usize);
        if UNIT_POS == 0 {
            mem[0] = 0xDEADBEEFu32;
        }
        sync_cube();
        out[UNIT_POS as usize] = mem[0];
    }

    #[cube(launch)]
    fn sync_cube_two_phase(out: &mut [u32]) {
        let mut mem = Shared::new_slice(4usize);
        let idx = UNIT_POS as usize;
        mem[idx] = (idx as u32) + 1;
        sync_cube();

        if UNIT_POS == 0 {
            let mut sum = 0u32;
            for i in 0..4 {
                sum += mem[i];
            }
            mem[0] = sum;
        }
        sync_cube();

        out[idx] = mem[0];
    }

    // Two shared memories of different alignments must each be shared by the whole cube, and
    // must not overlap.
    #[cube(launch)]
    fn sync_cube_two_shared(out: &mut [u32]) {
        let mut units = Shared::new_slice(4usize);
        let mut scaled = Shared::new_slice(4usize);
        let idx = UNIT_POS as usize;
        units[idx] = UNIT_POS + 1;
        scaled[idx] = 10u64 * (UNIT_POS as u64 + 1);
        sync_cube();

        let mut sum = 0u32;
        for i in 0..4 {
            sum += units[i] + scaled[i] as u32;
        }
        out[idx] = sum;
    }

    #[cube(launch)]
    fn sync_cube_all_reduce(out: &mut [u32]) {
        let mut mem = Shared::new_slice(8usize);
        let idx = UNIT_POS as usize;
        mem[idx] = idx as u32;
        sync_cube();

        let mut sum = 0u32;
        for i in 0..8 {
            sum += mem[i];
        }
        out[idx] = sum;
    }

    // Reads an input into shared memory at a computed (non-identity) index, then reads it
    // back. If shared memory is reserved from the same pool as the input binding, the two
    // alias and `mem[j] = input[i]` corrupts the input in place.
    #[cube(launch)]
    fn shared_scatter_gather(input: &[f32], output: &mut [f32], #[comptime] n: usize) {
        let mut mem = Shared::new_slice(n);
        let mut i = 0usize;
        while i < n {
            mem[(i + 2) % n] = input[i];
            i += 1;
        }
        sync_cube();
        let mut k = 0usize;
        while k < n {
            output[k] = mem[k];
            k += 1;
        }
    }

    #[cube(launch_unchecked)]
    fn delayed_copy(input: &[u32], output: &mut [u32], num_loop: usize) {
        if UNIT_POS == 0 {
            let mut pos = 0usize;
            for i in 0..num_loop {
                pos = (pos + i) % input.len();
            }
            output[0] = input[pos];
        }
    }

    #[test]
    fn test_barrier_smoke_cpu() {
        let client = TestRuntime::client(&Default::default());
        let out = client.empty(core::mem::size_of::<f32>());

        unsafe {
            barrier_smoke::launch::<TestRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(out.clone(), 1),
            )
        }

        let bytes = client.read_one_unchecked(out);
        let actual = f32::from_bytes(&bytes);
        assert_eq!(actual[0], 1.0);
    }

    #[test]
    fn test_sync_cube_magic_cpu() {
        let client = TestRuntime::client(&Default::default());
        let out = client.empty(4 * core::mem::size_of::<u32>());

        unsafe {
            sync_cube_magic::launch::<TestRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(4),
                BufferArg::from_raw_parts(out.clone(), 4),
            )
        }

        let bytes = client.read_one_unchecked(out);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[0xDEADBEEF; 4]);
    }

    #[test]
    fn test_sync_cube_two_phase_cpu() {
        let client = TestRuntime::client(&Default::default());
        let out = client.empty(4 * core::mem::size_of::<u32>());

        unsafe {
            sync_cube_two_phase::launch::<TestRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(4),
                BufferArg::from_raw_parts(out.clone(), 4),
            )
        }

        let bytes = client.read_one_unchecked(out);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[10u32; 4]);
    }

    #[test]
    fn test_sync_cube_two_shared_cpu() {
        let client = TestRuntime::client(&Default::default());
        let out = client.empty(4 * core::mem::size_of::<u32>());

        unsafe {
            sync_cube_two_shared::launch::<TestRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(4),
                BufferArg::from_raw_parts(out.clone(), 4),
            )
        }

        let bytes = client.read_one_unchecked(out);
        let actual = u32::from_bytes(&bytes);
        // (1 + 2 + 3 + 4) + (10 + 20 + 30 + 40)
        assert_eq!(actual, &[110u32; 4]);
    }

    #[test]
    fn test_sync_cube_all_reduce_cpu() {
        let client = TestRuntime::client(&Default::default());
        let out = client.empty(8 * core::mem::size_of::<u32>());

        unsafe {
            sync_cube_all_reduce::launch::<TestRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(8),
                BufferArg::from_raw_parts(out.clone(), 8),
            )
        }

        let bytes = client.read_one_unchecked(out);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[28u32; 8]);
    }

    #[test]
    fn shared_memory_does_not_alias_input_binding() {
        let client = TestRuntime::client(&Default::default());
        let n = 8usize;
        let input = client.create_from_slice(f32::as_bytes(&[
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        ]));
        let out = client.empty(n * core::mem::size_of::<f32>());

        unsafe {
            shared_scatter_gather::launch::<TestRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(input, n),
                BufferArg::from_raw_parts(out.clone(), n),
                n,
            )
        }

        let bytes = client.read_one_unchecked(out);
        let actual = f32::from_bytes(&bytes);
        // output[k] = input[(k + n - 2) % n]
        let expected: Vec<f32> = (0..n).map(|k| (10 + (k + n - 2) % n) as f32).collect();
        assert_eq!(actual, expected.as_slice());
    }

    #[test]
    fn queued_cpu_kernel_keeps_buffer_bindings_alive_until_execution() {
        let client = TestRuntime::client(&Default::default());
        let max_streams = CubeClRuntimeConfig::get().streaming.max_streams as u64;

        let stream_a = StreamId { value: 0 };
        let stream_b = StreamId { value: max_streams };

        let client_a = unsafe {
            let mut client = client.clone();
            client.set_stream(stream_a);
            client
        };
        let client_b = unsafe {
            let mut client = client.clone();
            client.set_stream(stream_b);
            client
        };

        let input = client_a.create_from_slice(u32::as_bytes(&[7, 7]));
        let output = client_a.empty(core::mem::size_of::<u32>());

        unsafe {
            delayed_copy::launch_unchecked::<TestRuntime>(
                &client_a,
                CubeCount::new_single(),
                CubeDim::new_1d(1),
                BufferArg::from_raw_parts(input, 2),
                BufferArg::from_raw_parts(output.clone(), 1),
                5_000_001,
            )
        }

        let replacement = client_b.create_from_slice(u32::as_bytes(&[99, 99]));
        drop(replacement);

        let bytes = client_a.read_one_unchecked(output);
        let actual = u32::from_bytes(&bytes);
        assert_eq!(actual, &[7]);
    }

    /// The stream error queue is per logical stream on the CPU as on every
    /// device backend.
    ///
    /// Logical streams are folded onto the pooled ones with `id % max_streams`,
    /// so two of them share a backend stream. A neighbour that drained the
    /// rejection would fail on a kernel it never launched, while the stream
    /// that did launch it read back an untouched buffer as if all was well.
    mod stream_errors {
        use cubecl_core::prelude::*;
        use cubecl_core::{self as cubecl};
        use cubecl_environment::stream::StreamId;
        use cubecl_runtime::{
            config::{CubeClRuntimeConfig, RuntimeConfig},
            server::Handle,
        };

        use super::TestRuntime;

        /// A launch the compiler is guaranteed to refuse, whatever the target.
        #[cube(launch_unchecked)]
        fn rejected(out: &mut [u32], #[comptime] reason: String) {
            push_validation_error(reason);
            out[0] = 1u32;
        }

        /// Two logical streams landing on one pooled stream.
        ///
        /// `seed` is far above the ids [`StreamId::current`] hands out per
        /// thread, which are small and sequential: a test that pins a low id
        /// shares it outright with whichever sibling test's thread was assigned
        /// the same number, and then legitimately drains that sibling's errors.
        fn sharing_one_pooled_stream(seed: u64) -> (StreamId, StreamId) {
            let max_streams = CubeClRuntimeConfig::get().streaming.max_streams as u64;
            (
                StreamId { value: seed },
                StreamId {
                    value: seed + max_streams,
                },
            )
        }

        fn launch_rejected(client: &ComputeClient<TestRuntime>, reason: &str) -> Handle {
            let out = client.empty(core::mem::size_of::<u32>());
            unsafe {
                rejected::launch_unchecked::<TestRuntime>(
                    &client.clone(),
                    CubeCount::new_single(),
                    CubeDim::new_1d(1),
                    BufferArg::from_raw_parts(out.clone(), 1),
                    reason.to_string(),
                )
            };
            out
        }

        fn assert_rejected(client: &ComputeClient<TestRuntime>, out: Handle, reason: &str) {
            let err = client
                .read_one(out)
                .expect_err("the kernel pushed a validation error, the launch must fail")
                .to_string();
            assert!(
                err.contains(reason),
                "the read must report the launch that never wrote the buffer, got: {err}"
            );
        }

        /// A rejected launch belongs to the stream that made it, and to no
        /// neighbour sharing its pooled stream.
        #[test]
        fn a_rejected_launch_stays_on_its_own_stream() {
            let client = TestRuntime::client(&Default::default());
            let (launching, neighbour) = sharing_one_pooled_stream(1_000_001);

            let out = launching.executes(|| launch_rejected(&client, "cpu-attribution"));

            neighbour.executes(|| {
                client
                    .flush()
                    .expect("the neighbouring stream launched nothing, so it has nothing to report")
            });

            launching.executes(|| assert_rejected(&client, out, "cpu-attribution"));
        }

        /// A read is only as good as the work that wrote the buffer.
        ///
        /// The rejection belongs to the stream that launched, so the reader's
        /// own flush never sees it — and a read that does not consult the
        /// producer hands back the buffer the failed launch never wrote.
        #[test]
        fn a_read_surfaces_the_rejection_of_the_stream_that_wrote_the_buffer() {
            let client = TestRuntime::client(&Default::default());
            let (producer, reader) = sharing_one_pooled_stream(1_000_002);

            let out = producer.executes(|| launch_rejected(&client, "cpu-producer"));

            reader.executes(|| assert_rejected(&client, out, "cpu-producer"));
            // Reading the producer's error does not take it: the stream that
            // made the launch still reports it itself.
            producer.executes(|| {
                client
                    .flush()
                    .expect_err("the launching stream keeps its own rejection")
            });
        }
    }
}

pub mod compute;
pub mod device;
pub mod frontend;
pub mod runtime;

pub use device::CpuDevice;
pub use runtime::*;
