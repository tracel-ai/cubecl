use cubecl_core::{CubeElement, Runtime, client::ComputeClient, prelude::Float};
use cubecl_std::tensor::TensorHandle;

use crate::tests::test_utils::Sample;

pub(crate) struct MatmulTestCase {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
}

impl MatmulTestCase {
    pub(crate) fn matmul_cpu<R: Runtime, F: Float + CubeElement>(
        &self,
        lhs: &TensorHandle<R, F>,
        rhs: &TensorHandle<R, F>,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> Vec<F> {
        let lhs_binding = &client.read_one_tensor(lhs.handle.clone().binding_with_meta(
            lhs.shape.clone(),
            lhs.strides.clone(),
            size_of::<F>(),
        ));
        let rhs_binding = &client.read_one_tensor(rhs.handle.clone().binding_with_meta(
            rhs.shape.clone(),
            rhs.strides.clone(),
            size_of::<F>(),
        ));

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
        TensorHandle::empty(client, vec![self.batch, self.m, self.n])
    }

    pub(crate) fn random_tensor<R: Runtime, F: Float + CubeElement + Sample>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        shape: Vec<usize>,
    ) -> TensorHandle<R, F> {
        F::sample::<R>(client, &shape, 999)
    }
}
