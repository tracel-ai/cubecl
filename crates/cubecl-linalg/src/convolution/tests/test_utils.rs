use std::fmt::Display;

use cubecl_core::{
    CubeElement, Feature, Runtime,
    client::ComputeClient,
    prelude::{Float, Numeric},
    server::{self},
};

use crate::{
    convolution::base::ConvolutionProblem,
    matmul::tests::{CastInto, Sample},
};

pub trait TestPrecision {
    type EG: Numeric + CubeElement + Display + CastInto<Self::ES> + Sample;
    type ES: Numeric + Display + CastInto<Self::EA>;
    type EA: Numeric + Display + CastInto<Self::EG>;

    fn assert_result<R: Runtime>(
        lhs: &[Self::EG],
        rhs: &[Self::EG],
        problem: &ConvolutionProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: server::Handle,
        shape: &[usize],
        strides: &[usize],
    );
}

impl<EG, ES> TestPrecision for (EG, ES)
where
    EG: Float + CubeElement + Display + CastInto<ES> + Sample,
    ES: Numeric + Display + CastInto<f32>,
    f32: CastInto<EG>,
{
    type EG = EG;
    type ES = ES;
    type EA = f32;

    fn assert_result<R: Runtime>(
        lhs: &[EG],
        rhs: &[EG],
        problem: &ConvolutionProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: server::Handle,
        shape: &[usize],
        strides: &[usize],
    ) {
        let maybe_f16 = client.properties().feature_enabled(Feature::Cmma {
            a: ES::as_elem_native().expect("To be a native type"),
            b: ES::as_elem_native().expect("To be a native type"),
            c: EG::as_elem_native().expect("To be a native type"),
            m: 16,
            k: 16,
            n: 16,
        });
        let maybe_tf32 = client.properties().feature_enabled(Feature::Cmma {
            a: ES::as_elem_native().expect("To be a native type"),
            b: ES::as_elem_native().expect("To be a native type"),
            c: EG::as_elem_native().expect("To be a native type"),
            m: 16,
            k: 8,
            n: 16,
        });

        // Need to compensate for the temporary conversion to f16/tf32
        let epsilon = match maybe_f16 || maybe_tf32 {
            true => 10e-6 / EG::EPSILON.to_f32().unwrap() * half::f16::EPSILON.to_f32(),
            false => 10e-6,
        };

        let expected = conv_cpu_reference::<Self>(lhs, rhs, problem)
            .into_iter()
            .map(|x| x.cast_into())
            .collect::<Vec<EG>>();

        if let Err(e) =
            assert_equals_approx::<R, EG>(client, out, shape, strides, &expected, epsilon)
        {
            panic!("{}", e);
        }
    }
}

/// Compares the content of a handle to a given slice of f32.
pub(crate) fn assert_equals_approx<R: Runtime, F: Float + CubeElement + Display>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: server::Handle,
    shape: &[usize],
    strides: &[usize],
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let actual = client.read_one_tensor(output.binding_with_meta(
        shape.to_vec(),
        strides.to_vec(),
        size_of::<F>(),
    ));
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
        let allowed_error = (epsilon * e.to_f32().unwrap()).max(epsilon);

        if f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()) >= allowed_error {
            return Err(format!(
                "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
                i,
                *a,
                *e,
                f32::abs(a.to_f32().unwrap() - e.to_f32().unwrap()),
                epsilon
            ));
        }
    }

    // return Err("".to_string());

    Ok(())
}

/// Solves a matmul problem with EG inputs, multiplied as ES and accumulated as EA.
///
/// This is a naive CPU implementation, very slow on large payloads,
/// not designed to be used for other purposes than testing.
pub(crate) fn conv_cpu_reference<P: TestPrecision>(
    lhs: &[P::EG],
    rhs: &[P::EG],
    problem: &ConvolutionProblem,
) -> Vec<P::EA>
where
{
    let n = problem.batches;
    let h = problem.shape[0];
    let w = problem.shape[1];
    let c = problem.channels;

    let out_h = problem.out_shape[0];
    let out_w = problem.out_shape[1];
    let out_channels = problem.n;

    let kh = problem.kernel_size[0] as usize;
    let kw = problem.kernel_size[1] as usize;

    let padding = &problem.padding;
    let stride = &problem.stride;
    let dilation = &problem.dilation;

    let lhs_stride_n = h * w * c;
    let lhs_stride_h = w * c;
    let lhs_stride_w = c;
    let lhs_stride_c = 1;

    let rhs_stride_out_c = kh * kw * c;
    let rhs_stride_kh = kw * c;
    let rhs_stride_kw = c;
    let rhs_stride_in_c = 1;

    let out_stride_n = out_h * out_w * out_channels;
    let out_stride_h = out_w * out_channels;
    let out_stride_w = out_channels;
    let out_stride_c = 1;

    let mut out = vec![P::EA::from_int(0); n * out_h * out_w * out_channels];

    for nth_batch in 0..n {
        let batch_in = nth_batch * lhs_stride_n;
        let batch_out = nth_batch * out_stride_n;

        for out_y in 0..out_h {
            let out_offset = batch_out + out_y * out_stride_h;

            for out_x in 0..out_w {
                let out_offset = out_offset + out_x * out_stride_w;

                for out_c in 0..out_channels {
                    let out_pos = out_offset + out_c * out_stride_c;
                    let weight_offset = out_c * rhs_stride_out_c;

                    let mut acc = P::EA::from_int(0);
                    for in_c in 0..c {
                        let in_offset = batch_in + in_c * lhs_stride_c;
                        let weight_offset = weight_offset + in_c * rhs_stride_in_c;

                        for ky in 0..kh {
                            let weight_offset = weight_offset + ky * rhs_stride_kh;

                            for kx in 0..kw {
                                let weight_pos = weight_offset + kx * rhs_stride_kw;

                                let in_y = out_y as i32 * stride[0] as i32
                                    + ky as i32 * dilation[0] as i32
                                    - padding[0];
                                let in_x = out_x as i32 * stride[1] as i32
                                    + kx as i32 * dilation[1] as i32
                                    - padding[1];

                                if in_y >= 0 && in_y < h as i32 && in_x >= 0 && in_x < w as i32 {
                                    let in_pos = in_offset
                                        + in_y as usize * lhs_stride_h
                                        + in_x as usize * lhs_stride_w;

                                    let value: P::ES = lhs[in_pos].cast_into();
                                    let weight: P::ES = rhs[weight_pos].cast_into();
                                    acc += (value * weight).cast_into();
                                }
                            }
                        }
                    }
                    out[out_pos] = acc;
                }
            }
        }
    }

    out
}
