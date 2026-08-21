#[macro_use]
extern crate derive_new;

extern crate alloc;

mod backend;
mod compiler;
mod compute;
mod device;
mod graphics;
mod runtime;

pub use compiler::base::*;
pub use compute::*;
pub use device::*;
pub use graphics::*;
pub use runtime::*;

#[cfg(feature = "spirv")]
pub use backend::vulkan;

#[cfg(all(feature = "msl", target_os = "macos"))]
pub use backend::metal;

#[cfg(all(test, not(feature = "spirv"), not(feature = "msl")))]
#[allow(unexpected_cfgs)]
mod tests {
    pub type TestRuntime = crate::WgpuRuntime;
    use half::f16;

    // Include 64-bit types (i64, u64) for WGSL as wgpu supports them. These don't exist on
    // native WebGPU however.
    //
    // Also include f16, this is an extension but supported by wgpu and WebGPU.
    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i32, i64], u32: [u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([flex32, f32, u32]);
    cubecl_std::testgen_quantized_view!(f32);

    /// WGSL packs fp8 four lanes to a `u32` and has no type for anything narrower. Rejecting
    /// that has to reach the caller: a panic on the device thread is caught there, logged as a
    /// warning, and the caller reads back a zeroed buffer as if the launch had succeeded.
    mod fp8_lanes {
        use cubecl_common::e4m3;
        use cubecl_core::prelude::*;
        use cubecl_core::{self as cubecl};
        use cubecl_environment::stream::StreamId;
        use cubecl_runtime::{
            config::{CubeClRuntimeConfig, RuntimeConfig},
            server::Handle,
        };

        use super::TestRuntime;

        /// A cast, which the minifloat lowering pass is what rejects.
        #[cube(launch_unchecked)]
        fn cast_fp8<N: Size>(input: &[Vector<f32, N>], out: &mut [Vector<f32, N>]) {
            if ABSOLUTE_POS < input.len() {
                let codes = Vector::<e4m3, N>::cast_from(input[ABSOLUTE_POS]);
                out[ABSOLUTE_POS] = Vector::cast_from(codes);
            }
        }

        /// No cast at all, so only the WGSL type printer ever sees the fp8.
        #[cube(launch_unchecked)]
        fn copy_fp8<N: Size>(input: &[Vector<e4m3, N>], out: &mut [Vector<e4m3, N>]) {
            if ABSOLUTE_POS < input.len() {
                out[ABSOLUTE_POS] = input[ABSOLUTE_POS];
            }
        }

        fn assert_rejected(client: &ComputeClient<TestRuntime>, out: Handle) {
            let err = client
                .read_one(out)
                .expect_err("two fp8 lanes have no WGSL representation, the launch must fail")
                .to_string();
            assert!(
                err.contains("fp8 on WGSL is packed 4 lanes to a u32"),
                "the packing rule has to be in the error the caller sees, got: {err}"
            );
        }

        #[test]
        fn cast_at_two_lanes_is_reported() {
            let client = TestRuntime::client(&Default::default());
            let input = client.create_from_slice(&[0u8; 64]);
            let out = client.empty(64);
            unsafe {
                cast_fp8::launch_unchecked::<TestRuntime>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(8),
                    2,
                    BufferArg::from_raw_parts(input, 16),
                    BufferArg::from_raw_parts(out.clone(), 16),
                )
            };
            assert_rejected(&client, out);
        }

        #[test]
        fn copy_at_two_lanes_is_reported() {
            let client = TestRuntime::client(&Default::default());
            let input = client.create_from_slice(&[0u8; 32]);
            let out = client.empty(32);
            unsafe {
                copy_fp8::launch_unchecked::<TestRuntime>(
                    &client,
                    CubeCount::new_single(),
                    CubeDim::new_1d(8),
                    2,
                    BufferArg::from_raw_parts(input, 32),
                    BufferArg::from_raw_parts(out.clone(), 32),
                )
            };
            assert_rejected(&client, out);
        }

        /// A rejected launch belongs to the stream that made it. Logical streams
        /// are folded onto the pooled ones with `id % max_streams`, so two of
        /// them share a backend stream — and used to share its error queue: the
        /// neighbour drained the rejection, failing on a kernel it never
        /// launched, while the stream that did launch it read back a zeroed
        /// buffer as if all was well.
        #[test]
        fn a_rejected_launch_stays_on_its_own_stream() {
            let client = TestRuntime::client(&Default::default());
            let max_streams = CubeClRuntimeConfig::get().streaming.max_streams as u64;
            // Two logical streams, one pooled stream between them.
            let launching = StreamId { value: 1 };
            let neighbour = StreamId {
                value: 1 + max_streams,
            };

            let out = launching.executes(|| {
                let input = client.create_from_slice(&[0u8; 32]);
                let out = client.empty(32);
                unsafe {
                    copy_fp8::launch_unchecked::<TestRuntime>(
                        &client,
                        CubeCount::new_single(),
                        CubeDim::new_1d(8),
                        2,
                        BufferArg::from_raw_parts(input, 32),
                        BufferArg::from_raw_parts(out.clone(), 32),
                    )
                };
                out
            });

            neighbour.executes(|| {
                client
                    .flush()
                    .expect("the neighbouring stream launched nothing, so it has nothing to report")
            });

            launching.executes(|| assert_rejected(&client, out));
        }
    }
}

#[cfg(all(test, feature = "spirv"))]
#[allow(unexpected_cfgs)]
mod tests_spirv {
    pub type TestRuntime = crate::WgpuRuntime;
    use cubecl_core::flex32;
    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, flex32, f32], i32: [i8, i16, i32, i64], u32: [u8, u16, u32, u64]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, flex32, f32, u32]);
    cubecl_std::testgen_quantized_view!(f16);
}

#[cfg(all(test, feature = "msl"))]
#[allow(unexpected_cfgs)]
mod tests_msl {
    pub type TestRuntime = crate::WgpuRuntime;
    use half::f16;

    cubecl_core::testgen_all!(f32: [f16, f32], i32: [i16, i32], u32: [u16, u32]);
    cubecl_std::testgen!();
    cubecl_std::testgen_tensor_identity!([f16, flex32, f32, u32]);
    cubecl_std::testgen_quantized_view!(f16);
}
