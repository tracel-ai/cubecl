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
    cubecl_core::testgen_profiling!();

    /// A kernel that brings its own WGSL, the way a template kernel downstream
    /// does: no representation for the runtime to read, only text for naga.
    mod precompiled {
        use cubecl_core::prelude::*;
        use cubecl_ir::{UIntKind, metadata::Info, settings::Dim3};
        use cubecl_server::kernel::{
            CubeKernel, KernelDefinition, KernelMetadata, PrecompiledSource,
        };
        use cubecl_server::runtime::Runtime;
        use cubecl_server::server::KernelArguments;

        use super::TestRuntime;

        const SOURCE: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(1)
fn double(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = data[id.x] * 2.0;
}
"#;

        struct Double;

        impl KernelMetadata for Double {
            fn id(&self) -> KernelId {
                KernelId::new::<Self>()
            }

            fn address_type(&self) -> ElemType {
                ElemType::UInt(UIntKind::U32)
            }
        }

        impl CubeKernel for Double {
            fn define(&self) -> KernelDefinition {
                let settings = KernelSettings::new(
                    Dim3::new_single(),
                    ExecutionMode::Checked,
                    AddressType::U32,
                );
                KernelDefinition {
                    body: Scope::root(settings.clone()),
                    settings,
                    info: Info::default(),
                }
            }

            fn source(&self) -> Option<PrecompiledSource> {
                Some(PrecompiledSource {
                    source: SOURCE.to_string(),
                    entrypoint_name: "double".to_string(),
                    lang: "wgsl",
                })
            }
        }

        #[test]
        fn a_hand_written_wgsl_kernel_launches() {
            let client = TestRuntime::client(&Default::default());
            let input = [1.0f32, 2.0, 3.0, 4.0];
            let handle = client.create_from_slice(bytemuck::cast_slice(&input));

            client.launch(
                Box::new(Double),
                CubeCount::Static(input.len() as u32, 1, 1),
                KernelArguments::new().with_buffer(handle.clone().binding()),
            );

            let bytes = client.read_one(handle).expect("the launch ran");
            let output: &[f32] = bytemuck::cast_slice(&bytes);
            assert_eq!(output, [2.0, 4.0, 6.0, 8.0]);
        }
    }

    /// WGSL packs fp8 four lanes to a `u32` and has no type for anything narrower. Rejecting
    /// that has to reach the caller: a panic on the device thread is caught there, logged as a
    /// warning, and the caller reads back a zeroed buffer as if the launch had succeeded.
    mod fp8_lanes {
        use cubecl_common::e4m3;
        use cubecl_core::prelude::*;
        use cubecl_core::{self as cubecl};
        use cubecl_server::runtime::Runtime;
        use cubecl_server::server::Handle;

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

        fn assert_rejected(client: &Client, out: Handle) {
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
                cast_fp8::launch_unchecked(
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
                copy_fp8::launch_unchecked(
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
    }

    cubecl_core::testgen_complex_validation!();
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
    cubecl_core::testgen_profiling!();
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
