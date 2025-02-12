use std::fmt::Display;

use cubecl_core::{
    client::ComputeClient,
    flex32,
    prelude::{Float, Numeric},
    server::Handle,
    tf32, CubeElement, Feature, Runtime,
};

use crate::{
    matmul::{
        components::{Ident, MatmulProblem, MatmulSelection, MatrixLayout},
        kernels::matmul::Algorithm,
        tests::cmma_matmul::matmul_test_launcher::strides,
    },
    tensor::TensorHandle,
};

pub trait TestPrecision {
    type EG: Numeric + CubeElement + Display + CastInto<Self::ES> + Sample;
    type ES: Numeric + CubeElement + Display + CastInto<Self::EA>;
    type EA: Numeric + CubeElement + Display + CastInto<Self::EG>;

    fn assert_result<R: Runtime>(
        lhs: &[Self::EG],
        rhs: &[Self::EG],
        problem: &MatmulProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: Handle,
    );

    // TODO: This is a temporary hack to not run some quantized matmul test during development.
    //       This avoids breaking the CI with incomplete implementations.
    //       Remove when quantization is fully supported.
    fn should_run<A: Algorithm<Selection = MatmulSelection>>(
        layouts: (MatrixLayout, MatrixLayout),
    ) -> bool;
}

impl<EG, ES> TestPrecision for (EG, ES)
where
    EG: Float + CubeElement + Display + CastInto<ES> + Sample,
    ES: Float + CubeElement + Display + CastInto<f32>,
    f32: CastInto<EG>,
{
    type EG = EG;
    type ES = ES;
    type EA = f32;

    fn assert_result<R: Runtime>(
        lhs: &[EG],
        rhs: &[EG],
        problem: &MatmulProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: Handle,
    ) {
        let maybe_cmma = client.properties().feature_enabled(Feature::Cmma {
            a: ES::as_elem_native().expect("To be a native type"),
            b: ES::as_elem_native().expect("To be a native type"),
            c: EG::as_elem_native().expect("To be a native type"),
            m: 16,
            k: 16,
            n: 16,
        });

        // Need to compensate for the temporary conversion to f16/tf32
        let epsilon = match maybe_cmma {
            true => 10e-5 / EG::EPSILON.to_f32().unwrap() * half::f16::EPSILON.to_f32(),
            false => 10e-5,
        };

        let expected = matmul_cpu_reference::<Self>(lhs, rhs, problem)
            .into_iter()
            .map(|x| x.cast_into())
            .collect::<Vec<EG>>();

        if let Err(e) = assert_equals_approx::<R, EG>(client, out, &expected, epsilon) {
            panic!("{}", e);
        }
    }

    fn should_run<A: Algorithm<Selection = MatmulSelection>>(
        _layouts: (MatrixLayout, MatrixLayout),
    ) -> bool {
        true
    }
}

/// Compares the content of a handle to a given slice of f32.
pub(crate) fn assert_equals_approx<R: Runtime, F: Float + CubeElement + Display>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[F],
    epsilon: f32,
) -> Result<(), String> {
    let actual = client.read_one(output.binding());
    let actual = F::from_bytes(&actual);

    // normalize to type epsilon
    let epsilon = (epsilon / f32::EPSILON * F::EPSILON.to_f32().unwrap()).max(epsilon);

    // println!("{:?}", expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        // account for lower precision at higher values
        let allowed_error = (epsilon * e.to_f32().unwrap()).max(epsilon);
        // println!("Index={:?} Actual={:?} Expected={:?}", i, a, e);

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

// TODO:
//   - Add different conversions from i32 to u8.
//   - Add support for multipliers (zero_offsets).
pub struct Quantized;

impl TestPrecision for Quantized {
    type EG = u8;
    type ES = u16;
    type EA = i32;

    fn assert_result<R: Runtime>(
        lhs: &[u8],
        rhs: &[u8],
        problem: &MatmulProblem,
        client: &ComputeClient<R::Server, R::Channel>,
        out: Handle,
    ) {
        let expected = matmul_cpu_reference_quantized(lhs, rhs, 1, 1, problem) // TODO: Use different zero_offsets != 1.
            .into_iter()
            .map(|x| x.cast_into()) // TODO: Improve with different conversions.
            .collect::<Vec<u8>>();
        let actual = client.read_one(out.binding());
        let actual = u8::from_bytes(&actual);
        assert_eq!(actual, expected);
    }

    fn should_run<A: Algorithm<Selection = MatmulSelection>>(
        _layouts: (MatrixLayout, MatrixLayout),
    ) -> bool {
        false
    }
}

pub trait CastInto<E> {
    fn cast_into(self) -> E;
}

impl<E> CastInto<E> for E {
    fn cast_into(self) -> E {
        self
    }
}

impl CastInto<f32> for half::f16 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<f32> for half::bf16 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<f32> for flex32 {
    fn cast_into(self) -> f32 {
        f32::from(self)
    }
}

impl CastInto<half::bf16> for f32 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self)
    }
}

impl CastInto<half::bf16> for half::f16 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self.to_f32())
    }
}

impl CastInto<half::f16> for half::bf16 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self.to_f32())
    }
}

impl CastInto<half::f16> for f32 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self)
    }
}

impl CastInto<half::f16> for flex32 {
    fn cast_into(self) -> half::f16 {
        half::f16::from_f32(self.to_f32())
    }
}

impl CastInto<half::bf16> for flex32 {
    fn cast_into(self) -> half::bf16 {
        half::bf16::from_f32(self.to_f32())
    }
}

impl CastInto<flex32> for f32 {
    fn cast_into(self) -> flex32 {
        flex32::from_f32(self)
    }
}

impl CastInto<u16> for u8 {
    fn cast_into(self) -> u16 {
        self as u16
    }
}

impl CastInto<i32> for u16 {
    fn cast_into(self) -> i32 {
        self as i32
    }
}

impl CastInto<u8> for i32 {
    fn cast_into(self) -> u8 {
        self as u8
    }
}

pub trait Sample: Sized {
    fn sample(num_elements: usize, seed: u64) -> Vec<Self>;
}

macro_rules! sample_float {
    ($($t:ty),*) => {
        $(
            impl Sample for $t
            {
                fn sample(num_elements: usize, mut seed: u64) -> Vec<Self> {
                    fn lcg(seed: &mut u64) -> f32 {
                        const A: u64 = 1664525;
                        const C: u64 = 1013904223;
                        const M: f64 = 2u64.pow(32) as f64;

                        *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % (1u64 << 32);
                        (*seed as f64 / M * 2.0 - 1.0) as f32
                    }

                    (0..num_elements).map(|_| <$t as Float>::new(lcg(&mut seed))).collect()
                }
            }
        )*
    };
}

sample_float!(half::f16);
sample_float!(half::bf16);
sample_float!(flex32);
sample_float!(f32);
sample_float!(tf32);
sample_float!(f64);

impl Sample for u8 {
    fn sample(num_elements: usize, mut seed: u64) -> Vec<Self> {
        fn lcg(seed: &mut u64) -> u8 {
            const A: u64 = 1664525;
            const C: u64 = 1013904223;
            const M: u64 = 2u64.pow(32);

            *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % M;
            (*seed % 4) as u8
        }

        (0..num_elements).map(|_| lcg(&mut seed)).collect()
    }
}

/// Solves a matmul problem with EG inputs, multiplied as ES and accumulated as EA.
///
/// This is a naive CPU implementation, very slow on large payloads,
/// not designed to be used for other purposes than testing.
pub(crate) fn matmul_cpu_reference<P: TestPrecision>(
    lhs: &[P::EG],
    rhs: &[P::EG],
    problem: &MatmulProblem,
) -> Vec<P::EA>
where
{
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;
    let num_batches = problem.num_batches();

    let (b_lhs, b_rhs) = problem.batches.clone();
    assert!(b_lhs.len() == b_rhs.len(), "Cpu reference only works with batches of equal length. Please pad the shortest one with ones at the beginning.");

    let lhs_strides = strides(problem, Ident::Lhs);
    let rhs_strides = strides(problem, Ident::Rhs);
    let out_strides = strides(problem, Ident::Out);

    let mut out = vec![P::EA::from_int(0); m * n * num_batches];

    for nth_batch in 0..num_batches {
        let batch_out = nth_batch * m * n;
        let mut batch_lhs = 0;
        let mut batch_rhs = 0;
        for b in 0..b_lhs.len() {
            let tmp = batch_out / out_strides[b];
            batch_lhs += tmp % b_lhs[b] * lhs_strides[b];
            batch_rhs += tmp % b_rhs[b] * rhs_strides[b];
        }

        for i in 0..m {
            for j in 0..n {
                for k_ in 0..k {
                    let lhs_index = i * k + k_;
                    let rhs_index = k_ * n + j;
                    let out_index = i * n + j;

                    let l: P::ES = lhs[batch_lhs + lhs_index].cast_into();
                    let r: P::ES = rhs[batch_rhs + rhs_index].cast_into();
                    let prod = l * r;

                    out[batch_out + out_index] += prod.cast_into();
                }
            }
        }
    }

    out
}

pub(crate) fn matmul_cpu_reference_quantized(
    lhs: &[u8],
    rhs: &[u8],
    lhs_zero_offset: i32,
    rhs_zero_offset: i32,
    problem: &MatmulProblem,
) -> Vec<i32>
where
{
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;
    let num_batches = problem.num_batches();

    let (b_lhs, b_rhs) = problem.batches.clone();
    assert!(b_lhs.len() == b_rhs.len(), "Cpu reference only works with batches of equal length. Please pad the shortest one with ones at the beginning.");

    let lhs_strides = strides(problem, Ident::Lhs);
    let rhs_strides = strides(problem, Ident::Rhs);
    let out_strides = strides(problem, Ident::Out);

    let mut out = vec![0; m * n * num_batches];

    for nth_batch in 0..num_batches {
        let batch_out = nth_batch * m * n;
        let mut batch_lhs = 0;
        let mut batch_rhs = 0;
        for b in 0..b_lhs.len() {
            let tmp = batch_out / out_strides[b];
            batch_lhs += tmp % b_lhs[b] * lhs_strides[b];
            batch_rhs += tmp % b_rhs[b] * rhs_strides[b];
        }

        // Perform matmul
        for row in 0..m {
            for col in 0..n {
                for middle in 0..k {
                    let lhs_index = row * k + middle;
                    let rhs_index = middle * n + col;
                    let out_index = row * n + col;

                    let l = lhs[batch_lhs + lhs_index] as u16;
                    let r = rhs[batch_rhs + rhs_index] as u16;
                    let prod = l * r;

                    out[batch_out + out_index] += prod as i32;
                }
            }
        }

        // Substract rhs_zero_offset * sum_rows(lhs)
        for row in 0..m {
            let mut sum = 0;
            for col in 0..k {
                sum += lhs[batch_lhs + row * k + col] as i32;
            }
            sum *= rhs_zero_offset;
            for col in 0..n {
                out[batch_out + row * n + col] -= sum;
            }
        }

        // Substract lhs_zero_offset * sum_cols(rhs)
        for col in 0..n {
            let mut sum = 0;
            for row in 0..k {
                sum += rhs[batch_lhs + row * n + col] as i32;
            }
            sum *= lhs_zero_offset;
            for row in 0..m {
                out[batch_out + row * n + col] -= sum;
            }
        }

        // Add final constant term
        for row in 0..m {
            for col in 0..n {
                out[batch_out + row * n + col] += (k as i32) * lhs_zero_offset * rhs_zero_offset;
            }
        }
    }

    out
}

/// Deprecated
pub(crate) struct MatmulTestCase {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
}

/// Deprecated
impl MatmulTestCase {
    pub(crate) fn matmul_cpu<R: Runtime, F: Float + CubeElement>(
        &self,
        lhs: &TensorHandle<R, F>,
        rhs: &TensorHandle<R, F>,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Vec<F> {
        let lhs_binding = &client.read_one(lhs.handle.clone().binding());
        let rhs_binding = &client.read_one(rhs.handle.clone().binding());

        let lhs = F::from_bytes(lhs_binding);
        let rhs = F::from_bytes(rhs_binding);

        self.matmul_cpu_algorithm(lhs, rhs)
    }

    fn matmul_cpu_algorithm<F: Float + CubeElement>(&self, lhs: &[F], rhs: &[F]) -> Vec<F> {
        let mut out = vec![F::from_int(0); self.batch * self.m * self.n];
        let lhs_batch_offset = self.m * self.k;
        let rhs_batch_offset = self.k * self.n;
        let out_batch_offset = self.m * self.n;

        for b in 0..self.batch {
            for i in 0..self.m {
                for j in 0..self.n {
                    for k_ in 0..self.k {
                        let lhs_value = lhs[b * lhs_batch_offset + i * self.k + k_];
                        let rhs_value = rhs[b * rhs_batch_offset + j + k_ * self.n];

                        let result = lhs_value * rhs_value;

                        out[b * out_batch_offset + i * self.n + j] += result;
                    }
                }
            }
        }

        out
    }

    pub(crate) fn random_lhs<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        self.random_tensor(client, vec![self.batch, self.m, self.k])
    }

    pub(crate) fn random_rhs<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        self.random_tensor(client, vec![self.batch, self.k, self.n])
    }

    pub(crate) fn empty_out<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        TensorHandle::new_contiguous(
            vec![self.batch, self.m, self.n],
            self.create_empty::<R>(client, self.batch * self.m, self.n),
        )
    }

    pub(crate) fn random_tensor<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        shape: Vec<usize>,
    ) -> TensorHandle<R, F> {
        let data = F::sample(shape.iter().product(), 999);
        let handle = client.create(bytemuck::cast_slice(&data));
        TensorHandle::new_contiguous(shape, handle)
    }

    pub(crate) fn create_empty<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        x: usize,
        y: usize,
    ) -> Handle {
        client.empty(x * y * core::mem::size_of::<f32>())
    }
}
