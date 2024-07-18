use cubecl_core::{frontend::F32, CubeElement, Runtime};
use half::f16;

use crate::{
    matmul::{cmma::matmul_cmma, tiling2d::matmul_tiling_2d},
    tensor::TensorHandle,
};

use super::test_utils::{
    assert_equals_approx, cmma_available, create_empty, range_tensor_with_factor,
};

pub fn test_matmul_cmma_one_cube<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
        factor: 100000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_cmma::<R>(device);
}

pub fn test_matmul_cmma_several_cubes<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
        factor: 100000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_cmma::<R>(device);
}

pub fn test_matmul_cmma_with_check_bounds<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
        factor: 1000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_cmma::<R>(device);
}

pub fn test_matmul_cmma_with_batches<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
        factor: 10000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_cmma::<R>(device);
}

pub fn test_matmul_tiling2d_one_cube<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 1,
        factor: 100000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_tiling2d::<R>(device);
}

pub fn test_matmul_tiling2d_several_cubes<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 256,
        k: 256,
        n: 256,
        batch: 1,
        factor: 100000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_tiling2d::<R>(device);
}

pub fn test_matmul_tiling2d_with_check_bounds<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 60,
        k: 60,
        n: 60,
        batch: 1,
        factor: 1000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_tiling2d::<R>(device);
}

pub fn test_matmul_tiling2d_with_batches<R: Runtime>(device: &R::Device) {
    MatmulTestCase {
        m: 64,
        k: 64,
        n: 64,
        batch: 3,
        factor: 10000.,
        epsilon: 0.1,
        compute_f16: true,
    }
    .test_tiling2d::<R>(device);
}

struct MatmulTestCase {
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    factor: f32,
    epsilon: f32,
    compute_f16: bool,
}

impl MatmulTestCase {
    fn test_tiling2d<R: Runtime>(&self, device: &R::Device) {
        let tensor_1 =
            range_tensor_with_factor::<R>(self.batch, self.m, self.k, self.factor, device);
        let tensor_2 =
            range_tensor_with_factor::<R>(self.batch, self.k, self.n, self.factor, device);
        let out = TensorHandle::new_contiguous(
            vec![self.batch, self.m, self.n],
            create_empty::<R>(self.batch * self.m, self.n, device),
        );

        let expected = self.matmul_cpu(
            f32::from_bytes(&R::client(device).read(tensor_1.handle.clone().binding())),
            f32::from_bytes(&R::client(device).read(tensor_2.handle.clone().binding())),
        );

        let out = matmul_tiling_2d::<R, F32>(tensor_1, tensor_2, out, Default::default(), device);

        assert_equals_approx::<R>(out.handle, &expected, self.epsilon, device);
    }

    fn test_cmma<R: Runtime>(&self, device: &R::Device) {
        if !cmma_available::<R>(device) {
            // We can't execute the test, skip.
            return;
        }

        let tensor_1 =
            range_tensor_with_factor::<R>(self.batch, self.m, self.k, self.factor, device);
        let tensor_2 =
            range_tensor_with_factor::<R>(self.batch, self.k, self.n, self.factor, device);
        let out = TensorHandle::new_contiguous(
            vec![self.batch, self.m, self.n],
            create_empty::<R>(self.batch * self.m, self.n, device),
        );

        let expected = self.matmul_cpu(
            f32::from_bytes(&R::client(device).read(tensor_1.handle.clone().binding())),
            f32::from_bytes(&R::client(device).read(tensor_2.handle.clone().binding())),
        );

        let out = matmul_cmma::<R, F32>(tensor_1, tensor_2, out, device);

        assert_equals_approx::<R>(out.handle, &expected, self.epsilon, device);
    }

    fn matmul_cpu(&self, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
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

                        let result = if self.compute_f16 {
                            (f16::from_f32(lhs_value) * f16::from_f32(rhs_value)).to_f32()
                        } else {
                            lhs_value * rhs_value
                        };

                        out[b * out_batch_offset + i * self.n + j] += result;
                    }
                }
            }
        }
        out
    }
}
