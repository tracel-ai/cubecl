use cubecl_common::quant::scheme::QuantParam;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::quant::round::round_up_to_param;

#[cube(launch_unchecked)]
fn kernel_round_up<F: Float>(input: &[F], out: &mut [F], #[comptime] param: QuantParam) {
    if ABSOLUTE_POS < input.len() {
        out[ABSOLUTE_POS] = round_up_to_param::<F>(input[ABSOLUTE_POS], param);
    }
}

/// The device rule has to agree with [`QuantParam::round_up`] exactly. They are separate
/// implementations of one policy, and a tensor quantized on one backend has to reconstruct the
/// same on another, so nothing but a differential check pins them together.
///
/// No capability gate: the kernel is instantiated at `f32` and `param` only selects comptime
/// constants, so the storage type is never named on device.
pub fn test_round_up_matches_host<R: Runtime>(client: ComputeClient<R>, param: QuantParam) {
    // Reaches below every param's minimum normal, where the spacing stops halving.
    let mut scales: Vec<f32> = (-26..8)
        .flat_map(|exp| (1..17).map(move |step| (step as f32 / 16.0) * 2f32.powi(exp)))
        .collect();
    // Reaches the saturation branch and the last steps before it, where stepping up runs
    // against the param's NaN or infinity encodings.
    let max = param.max_representable();
    scales.extend([max * 0.75, max * 0.999, max, max * 2.0, f32::MAX]);

    let actual = launch::<R>(&client, &scales, param);

    let mut bad = vec![];
    for (i, &scale) in scales.iter().enumerate() {
        let expected = param
            .round_up(scale)
            .expect("the device rule has no answer for this param either");
        if actual[i] != expected {
            bad.push(format!(
                "  {scale:e}: device {:e}, host {expected:e}",
                actual[i]
            ));
        }
    }
    if !bad.is_empty() {
        panic!(
            "{param:?}: {} of {} disagree\n{}",
            bad.len(),
            scales.len(),
            bad.join("\n")
        );
    }
}

fn launch<R: Runtime>(client: &ComputeClient<R>, scales: &[f32], param: QuantParam) -> Vec<f32> {
    let handle_in = client.create_from_slice(f32::as_bytes(scales));
    let handle_out = client.empty(size_of_val(scales));

    // WebGPU only guarantees 256 units per cube, so spread the values over cubes instead.
    let cube_dim = 256u32;

    unsafe {
        kernel_round_up::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static((scales.len() as u32).div_ceil(cube_dim), 1, 1),
            CubeDim::new_1d(cube_dim),
            BufferArg::from_raw_parts(handle_in, scales.len()),
            BufferArg::from_raw_parts(handle_out.clone(), scales.len()),
            param,
        )
    };

    f32::from_bytes(&client.read_one_unchecked(handle_out)).to_vec()
}

#[macro_export]
macro_rules! testgen_round {
    () => {
        $crate::testgen_round!(
            round_up_matches_host_f16 => F16,
            round_up_matches_host_bf16 => BF16,
            round_up_matches_host_ue4m3 => UE4M3,
            round_up_matches_host_f32 => F32
        );
    };
    ($($name:ident => $param:ident),*) => {
        mod round {
            use super::*;
            use cubecl_common::quant::scheme::QuantParam;

            $(
                #[$crate::tests::test_log::test]
                fn $name() {
                    let client = TestRuntime::client(&Default::default());
                    cubecl_std::tests::round::test_round_up_matches_host::<TestRuntime>(
                        client,
                        QuantParam::$param,
                    );
                }
            )*
        }
    };
}
