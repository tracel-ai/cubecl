use cubecl_core::CubeElement;
use cubecl_core::{client::ComputeClient, Runtime};
use half::f16;

use crate::tensor::TensorHandle;

use super::test_utils::{create_empty, random_tensor};

pub(crate) struct MatmulTestCase {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub batch: usize,
}

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

                        let result = (f16::from_f32(lhs_value) * f16::from_f32(rhs_value)).to_f32();

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
        random_tensor(client, vec![self.batch, self.m, self.k])
    }

    pub(crate) fn random_rhs<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, f32> {
        random_tensor(client, vec![self.batch, self.k, self.n])
    }

    pub(crate) fn empty_out<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
    ) -> TensorHandle<R, f32> {
        TensorHandle::new_contiguous(
            vec![self.batch, self.m, self.n],
            create_empty::<R>(client, self.batch * self.m, self.n),
        )
    }
}
