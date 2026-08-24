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

    #[cube(launch_unchecked)]
    fn transcendentals<N: Size>(
        input: &[Vector<f32, N>],
        exp: &mut [Vector<f32, N>],
        ln: &mut [Vector<f32, N>],
        sin: &mut [Vector<f32, N>],
        cos: &mut [Vector<f32, N>],
        tanh: &mut [Vector<f32, N>],
    ) {
        if ABSOLUTE_POS < input.len() {
            let x = input[ABSOLUTE_POS];
            exp[ABSOLUTE_POS] = x.exp();
            ln[ABSOLUTE_POS] = x.ln();
            sin[ABSOLUTE_POS] = x.sin();
            cos[ABSOLUTE_POS] = x.cos();
            tanh[ABSOLUTE_POS] = x.tanh();
        }
    }

    /// `[exp, ln, sin, cos, tanh]` of `input`, evaluated at `width` lanes to a vector.
    ///
    /// A width above one is what selects the polynomial polyfill over the target's own
    /// intrinsic, so the two widths answer different questions of the same kernel.
    fn transcendentals_of(input: &[f32], width: usize) -> [Vec<f32>; 5] {
        assert_eq!(
            input.len() % width,
            0,
            "a {width}-wide launch would leave the last {} of {} inputs unwritten",
            input.len() % width,
            input.len()
        );

        let client = TestRuntime::client(&Default::default());
        let n = input.len();
        let handle = client.create_from_slice(f32::as_bytes(input));
        let out: Vec<_> = (0..5)
            .map(|_| client.empty(n * core::mem::size_of::<f32>()))
            .collect();

        unsafe {
            transcendentals::launch_unchecked::<TestRuntime>(
                &client,
                CubeCount::new_single(),
                CubeDim::new_1d((n / width) as u32),
                width,
                BufferArg::from_raw_parts(handle, n),
                BufferArg::from_raw_parts(out[0].clone(), n),
                BufferArg::from_raw_parts(out[1].clone(), n),
                BufferArg::from_raw_parts(out[2].clone(), n),
                BufferArg::from_raw_parts(out[3].clone(), n),
                BufferArg::from_raw_parts(out[4].clone(), n),
            )
        }

        let read = |handle: &cubecl_runtime::server::Handle| {
            f32::from_bytes(&client.read_one_unchecked(handle.clone())).to_vec()
        };
        [
            read(&out[0]),
            read(&out[1]),
            read(&out[2]),
            read(&out[3]),
            read(&out[4]),
        ]
    }

    /// A NaN must arrive as a NaN, an infinity with its sign, and a zero with its sign.
    /// A tolerance on the difference sees none of the three: it cannot tell a NaN from a
    /// large wrong number, and `+0.0` and `-0.0` differ by nothing at all.
    #[track_caller]
    fn assert_matches_library(op: &str, x: f32, actual: f32, expected: f32) {
        let agrees = if expected.is_nan() {
            actual.is_nan()
        } else if expected.is_infinite() || expected == 0.0 {
            actual.to_bits() == expected.to_bits()
        } else {
            (actual - expected).abs() <= 1e-6 * expected.abs().max(1e-6)
        };
        assert!(
            agrees,
            "{op}({x:e}) gave {actual:e}, the library gives {expected:e}"
        );
    }

    /// Zero, the infinities, a NaN and the subnormals, which an accuracy sweep never lands
    /// on and where the polynomials read an exponent field that means none of the things it
    /// usually means.
    #[test]
    fn transcendentals_match_the_library_at_the_edges() {
        // Lengths a multiple of the widest line under test, or its tail goes unwritten.
        let finite_angles = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            3.5,
            -3.5,
            100.0,
            -100.0,
            1e-40,
            f32::MIN_POSITIVE,
            REDUCTION_LIMIT,
            -REDUCTION_LIMIT,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            6.5,
        ];
        let magnitudes = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            1e-40,
            f32::MIN_POSITIVE,
            1000.0,
            -1000.0,
            1e30,
            -1e30,
            88.0,
            -88.0,
            1e-8,
            10.0,
            -10.0,
            1e-30,
            0.25,
        ];

        for width in [1, 2, 4, 8] {
            let [exp, ln, _, _, tanh] = transcendentals_of(&magnitudes, width);
            for (i, &x) in magnitudes.iter().enumerate() {
                assert_matches_library("exp", x, exp[i], x.exp());
                assert_matches_library("ln", x, ln[i], x.ln());
                assert_matches_library("tanh", x, tanh[i], x.tanh());
            }

            let [_, _, sin, cos, _] = transcendentals_of(&finite_angles, width);
            for (i, &x) in finite_angles.iter().enumerate() {
                assert_matches_library("sin", x, sin[i], x.sin());
                assert_matches_library("cos", x, cos[i], x.cos());
            }
        }
    }

    /// The largest angle the three-part range reduction still means something for. Past it
    /// the polyfill says NaN rather than returning a number of the right magnitude and the
    /// wrong sign.
    const REDUCTION_LIMIT: f32 = (1u32 << 20) as f32;

    /// Past the limit the answer is unknown, and only the lined path is asked to say so:
    /// a scalar keeps the target's own routine, which the gate leaves untouched.
    #[test]
    fn a_lined_sin_beyond_the_reduction_limit_is_not_a_number() {
        let beyond = [2.0 * REDUCTION_LIMIT, -2.0 * REDUCTION_LIMIT, 1e7, 1e30];

        for width in [2, 4] {
            let [_, _, sin, cos, _] = transcendentals_of(&beyond, width);
            for (i, &x) in beyond.iter().enumerate() {
                assert!(
                    sin[i].is_nan(),
                    "sin({x:e}) at width {width} gave {:e}",
                    sin[i]
                );
                assert!(
                    cos[i].is_nan(),
                    "cos({x:e}) at width {width} gave {:e}",
                    cos[i]
                );
            }
        }

        let [_, _, sin, cos, _] = transcendentals_of(&beyond, 1);
        for (i, &x) in beyond.iter().enumerate() {
            assert!(sin[i].is_finite(), "scalar sin({x:e}) gave {:e}", sin[i]);
            assert!(cos[i].is_finite(), "scalar cos({x:e}) gave {:e}", cos[i]);
        }
    }

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
}

pub mod compute;
pub mod device;
pub mod frontend;
pub mod runtime;

pub use device::CpuDevice;
pub use runtime::*;
