#[macro_use]
extern crate derive_new;

extern crate alloc;

#[cfg(test)]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::CpuRuntime;

    pub use half::f16;

    use cubecl_common::{config::RuntimeConfig, stream_id::StreamId};
    use cubecl_core as cubecl;
    use cubecl_core::prelude::*;
    use cubecl_runtime::config::CubeClRuntimeConfig;

    cubecl_core::testgen_all!(f32: [f16, f32, f64], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, f32, u32]);
    cubecl_std::testgen_tensor_into_contiguous!();
    cubecl_std::testgen_quantized_view!(f32);
    cubecl_core::testgen_complex_validation!();

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
}

pub mod compiler;
pub mod compute;
pub mod device;
pub mod frontend;
pub mod runtime;

pub use device::CpuDevice;
pub use runtime::*;
