use std::fmt::Display;

use cubecl_core::{
    client::ComputeClient,
    flex32,
    prelude::{Float, Numeric},
    server::Handle,
    CubeElement, Runtime,
};

use crate::{matmul::components::MatmulProblem, tensor::TensorHandle};

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

    Ok(())
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

/// Generates num_elements random floats for tests.
///
/// This is a naive CPU implementation with fixed seed,
/// not designed to be used for other purposes than testing.
pub(crate) fn generate_random_data<F: Float + CubeElement>(
    num_elements: usize,
    mut seed: u64,
) -> Vec<F> {
    fn lcg(seed: &mut u64) -> f32 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: f64 = 2u64.pow(32) as f64;

        *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % (1u64 << 32);
        (*seed as f64 / M * 2.0 - 1.0) as f32
    }

    (0..num_elements).map(|_| F::new(lcg(&mut seed))).collect()
}

/// Solves a matmul problem with EG inputs, multiplied as ES
///
/// This is a naive CPU implementation, very slow on large payloads,
/// not designed to be used for other purposes than testing.
pub(crate) fn matmul_cpu_with_broadcasting<EG, ES>(
    lhs: &[EG],
    rhs: &[EG],
    problem: &MatmulProblem,
) -> Vec<EG>
where
    EG: Numeric + CubeElement + CastInto<ES>,
    ES: Numeric + CubeElement + CastInto<EG>,
{
    let m = problem.m;
    let n = problem.n;
    let k = problem.k;
    let b = problem.num_batches();
    let mut out = vec![EG::from_int(0); m * n * b];

    for b_ in 0..b {
        let lhs_batch_stride = if problem.lhs_broadcast_batch { 0 } else { m * k };
        let rhs_batch_stride = if problem.rhs_broadcast_batch { 0 } else { k * n };

        let lhs_batch_start = b_ * lhs_batch_stride;
        let rhs_batch_start = b_ * rhs_batch_stride;

        for i in 0..m {
            for j in 0..n {
                for k_ in 0..k {
                    let lhs_index = lhs_batch_start + 
                        if problem.lhs_broadcast_batch { 0 } else { i * k + k_ };
                    let rhs_index = rhs_batch_start + 
                        if problem.rhs_broadcast_batch { 0 } else { k_ * n + j };
                    let out_index = b_ * m * n + i * n + j;

                    let l: ES = lhs[lhs_index].cast_into();
                    let r: ES = rhs[rhs_index].cast_into();
                    let prod = l * r;
                    let casted: EG = prod.cast_into();
                    out[out_index] += casted;
                }
            }
        }
    }
    out
}
// pub(crate) fn matmul_cpu_reference<EG, ES>(
//     lhs: &[EG],
//     rhs: &[EG],
//     problem: &MatmulProblem,
// ) -> Vec<EG>
// where
//     EG: Numeric + CubeElement + CastInto<ES>,
//     ES: Numeric + CubeElement + CastInto<EG>,
// {
//     let m = problem.m;
//     let n = problem.n;
//     let k = problem.k;
//     let b = problem.num_batches();

//     let mut out = vec![EG::from_int(0); m * n * b];

//     for b_ in 0..b {
//         for i in 0..m {
//             for j in 0..n {
//                 for k_ in 0..k {
//                     let lhs_index = b_ * m * k + i * k + k_;
//                     let rhs_index = b_ * k * n + k_ * n + j;
//                     let out_index = b_ * m * n + i * n + j;

//                     let l: ES = lhs[lhs_index].cast_into();
//                     let r: ES = rhs[rhs_index].cast_into();
//                     let prod = l * r;
//                     let casted: EG = prod.cast_into();

//                     out[out_index] += casted;
//                 }
//             }
//         }
//     }

//     out
// }

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

    pub(crate) fn random_lhs<R: Runtime, F: Float + CubeElement>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        self.random_tensor(client, vec![self.batch, self.m, self.k])
    }

    pub(crate) fn random_rhs<R: Runtime, F: Float + CubeElement>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        self.random_tensor(client, vec![self.batch, self.k, self.n])
    }

    pub(crate) fn empty_out<R: Runtime, F: Float + CubeElement>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, F> {
        TensorHandle::new_contiguous(
            vec![self.batch, self.m, self.n],
            self.create_empty::<R>(client, self.batch * self.m, self.n),
        )
    }

    pub(crate) fn random_tensor<R: Runtime, F: Float + CubeElement>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        shape: Vec<usize>,
    ) -> TensorHandle<R, F> {
        let data = generate_random_data::<F>(shape.iter().product(), 999);
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
