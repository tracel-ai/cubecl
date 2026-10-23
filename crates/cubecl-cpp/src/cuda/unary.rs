use cubecl_core::ir::{dialect::math::TanhOp, prelude::*};

use crate::{
    shared::CompilationOptions,
    shared::ty::TypedExtCPP,
    shared::unary::{lower_target_unop, tanh_via_f32},
    target::Cuda,
};

// CUDA headers before 12.8 lack htanh and h2tanh. Lower before packing.
lower_target_unop!(TanhOp, tanh_via_f32, Cuda, |op, ctx| {
    op.get_result(ctx).is_half(ctx)
        && !ctx
            .aux_ty::<CompilationOptions>()
            .supports_features
            .fast_tanh
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::CppCompiler;
    use cubecl_core as cubecl;
    use cubecl_core::ir::{
        DeviceIdentity, HardwareProperties, MemoryDeviceProperties, features::Features,
    };
    use cubecl_core::prelude::*;
    use cubecl_runtime::{compiler::Compiler, kernel::CubeKernel};
    use half::{bf16, f16};
    use std::sync::Arc;

    #[cube(launch)]
    fn scalar_tanh(input: &[f16], output: &mut [f16]) {
        output[0] = input[0].tanh();
    }

    #[cube(launch)]
    fn pair_tanh(input: &[Vector<f16, Const<2>>], output: &mut [Vector<f16, Const<2>>]) {
        output[0] = input[0].tanh();
    }

    #[cube(launch)]
    fn bf16_pair_tanh(input: &[Vector<bf16, Const<2>>], output: &mut [Vector<bf16, Const<2>>]) {
        output[0] = input[0].tanh();
    }

    fn source(kernel: impl CubeKernel, fast_tanh: bool) -> String {
        let mut options = CompilationOptions::default();
        options.supports_features.fast_tanh = fast_tanh;
        CppCompiler::<Cuda>::default()
            .compile(kernel.define(), &options)
            .unwrap()
            .source
    }

    fn properties() -> Arc<DeviceProperties> {
        Arc::new(DeviceProperties::new(
            Features::default(),
            MemoryDeviceProperties::new(u64::MAX, 256),
            HardwareProperties {
                load_width: 128,
                vector_register_count: None,
                plane_size_min: 32,
                plane_size_max: 32,
                max_bindings: 32,
                max_shared_memory_size: 65536,
                max_cube_count: (u32::MAX, u16::MAX as u32, u16::MAX as u32),
                max_units_per_cube: 1024,
                max_cube_dim: (1024, 1024, 1024),
                num_streaming_multiprocessors: None,
                num_tensor_cores: None,
                min_tensor_cores_dim: None,
                num_cpu_cores: None,
                last_level_cache_size: None,
                max_vector_size: VectorSize::MAX,
                cube_mma_reserved_shared_memory: 0,
            },
            cubecl_core::profile::TimingMethod::Device,
            DeviceIdentity {
                name: "offline".to_string(),
                fingerprint: "offline".to_string(),
                physical: None,
            },
        ))
    }

    #[test]
    fn emitted_cuda_tanh_respects_toolkit_support_before_packing() {
        let settings = KernelSettings::new(
            *CubeDim::new_1d(1),
            ExecutionMode::Unchecked,
            AddressType::U32,
        );
        let scalar = || {
            scalar_tanh::ScalarTanh::new(
                settings.clone(),
                properties(),
                Arc::new(TargetProperties::default()),
                BufferCompilationArg { inplace: None },
                BufferCompilationArg { inplace: None },
            )
        };
        let pair = || {
            pair_tanh::PairTanh::new(
                settings.clone(),
                properties(),
                Arc::new(TargetProperties::default()),
                BufferCompilationArg { inplace: None },
                BufferCompilationArg { inplace: None },
            )
        };
        let bf16_pair = || {
            bf16_pair_tanh::Bf16PairTanh::new(
                settings.clone(),
                properties(),
                Arc::new(TargetProperties::default()),
                BufferCompilationArg { inplace: None },
                BufferCompilationArg { inplace: None },
            )
        };

        let scalar_fallback = source(scalar(), false);
        let pair_fallback = source(pair(), false);
        let bf16_fallback = source(bf16_pair(), false);
        for fallback in [scalar_fallback, pair_fallback, bf16_fallback] {
            assert!(fallback.contains("tanh("), "{fallback}");
            assert!(!fallback.contains("htanh("), "{fallback}");
            assert!(!fallback.contains("h2tanh("), "{fallback}");
        }

        let scalar_native = source(scalar(), true);
        let pair_native = source(pair(), true);
        let bf16_native = source(bf16_pair(), true);
        assert!(scalar_native.contains("htanh("), "{scalar_native}");
        assert!(pair_native.contains("h2tanh("), "{pair_native}");
        assert!(bf16_native.contains("h2tanh("), "{bf16_native}");
    }
}
