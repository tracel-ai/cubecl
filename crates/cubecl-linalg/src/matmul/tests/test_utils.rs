use cubecl_core::{client::ComputeClient, prelude::Numeric, server::Handle, CubeElement, Runtime};

use crate::{matmul::matmul_modular::problem::MatmulProblem, tensor::TensorHandle};

/// Compares the content of a handle to a given slice of f32.
pub(crate) fn assert_equals_approx<I: CubeElement, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    output: Handle,
    expected: &[f32],
    epsilon: f32,
) -> Result<(), String> {
    let actual = client.read(output.binding());
    let actual = I::from_bytes(&actual);
    println!("{:?}", actual);
    println!("{:?}", expected);

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let a = I::to_f32_value(*a);
        if (a - e).abs() >= epsilon {
            return Err(format!(
            "Values differ more than epsilon: index={} actual={}, expected={}, difference={}, epsilon={}",
            i,
            a,
            e,
            (a - e).abs(),
            epsilon
            ));
        }
    }

    Ok(())
}

/// Generates num_elements random f32 for tests.
///
/// This is a naive CPU implementation with fixed seed,
/// not designed to be used for other purposes than testing.
pub(crate) fn generate_random_data(num_elements: usize) -> Vec<f32> {
    fn lcg(seed: &mut u64) -> f32 {
        const A: u64 = 1664525;
        const C: u64 = 1013904223;
        const M: f64 = 2u64.pow(32) as f64;

        *seed = (A.wrapping_mul(*seed).wrapping_add(C)) % (1u64 << 32);
        (*seed as f64 / M * 2.0 - 1.0) as f32
    }

    let mut seed = 12345;

    (0..num_elements).map(|_| lcg(&mut seed)).collect()
}

/// Solves a matmul problem on f32 inputs.
///
/// This is a naive CPU implementation, very slow on large payloads,
/// not designed to be used for other purposes than testing.
pub(crate) fn matmul_cpu_reference<EG: Numeric>(
    lhs: &[f32],
    rhs: &[f32],
    problem: &MatmulProblem<EG>,
) -> Vec<f32> {
    let m = problem.m as usize;
    let n = problem.n as usize;
    let k = problem.k as usize;
    let b = problem.num_batches();

    let mut out = vec![0.; m * n * b];

    for b_ in 0..b {
        for i in 0..m {
            for j in 0..n {
                for k_ in 0..k {
                    out[(b_ * m * n) + i * n + j] +=
                        lhs[(b_ * m * k) + i * k + k_] * rhs[(b_ * k * n) + k_ * n + j];
                }
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
    pub(crate) fn matmul_cpu<R: Runtime>(
        &self,
        lhs: &TensorHandle<R, f32>,
        rhs: &TensorHandle<R, f32>,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Vec<f32> {
        let lhs_binding = &client.read(lhs.handle.clone().binding());
        let rhs_binding = &client.read(rhs.handle.clone().binding());

        let lhs = f32::from_bytes(lhs_binding);
        let rhs = f32::from_bytes(rhs_binding);

        self.matmul_cpu_algorithm(lhs, rhs)
    }

    fn matmul_cpu_algorithm(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        let mut out = vec![0.; self.batch * self.m * self.n];
        let lhs_batch_offset = self.m * self.k;
        let rhs_batch_offset = self.k * self.n;
        let out_batch_offset = self.m * self.n;

        for b in 0..self.batch {
            for i in 0..self.m {
                for j in 0..self.n {
                    for k_ in 0..self.k {
                        let lhs_value = lhs[b * lhs_batch_offset + i * self.k + k_];
                        let rhs_value = rhs[b * rhs_batch_offset + j + k_ * self.n];

                        let result = (half::f16::from_f32(lhs_value)
                            * half::f16::from_f32(rhs_value))
                        .to_f32();

                        out[b * out_batch_offset + i * self.n + j] += result;
                    }
                }
            }
        }

        out
    }

    pub(crate) fn random_lhs<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, f32> {
        self.random_tensor(client, vec![self.batch, self.m, self.k])
    }

    pub(crate) fn random_rhs<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, f32> {
        self.random_tensor(client, vec![self.batch, self.k, self.n])
    }

    pub(crate) fn empty_out<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, f32> {
        TensorHandle::new_contiguous(
            vec![self.batch, self.m, self.n],
            self.create_empty::<R>(client, self.batch * self.m, self.n),
        )
    }

    pub(crate) fn random_tensor<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        shape: Vec<usize>,
    ) -> TensorHandle<R, f32> {
        let data = generate_random_data(shape.iter().product());
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
