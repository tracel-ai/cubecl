use std::marker::PhantomData;

use crate::{matmul::test_utils::range_tensor_with_factor, tensor::Tensor};
use cubecl_core::{frontend::F32, CubeElement, Runtime};

use super::{
    cmma::matmul_cmma,
    test_utils::{assert_equals, create_empty},
};
use half::f16;

fn matmul_cpu(
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
    compute_f16: bool,
) -> Vec<f32> {
    let mut out = vec![0.; m * n];
    for i in 0..m {
        for j in 0..n {
            for k_ in 0..k {
                let lhs_value = lhs[i * k + k_];
                let rhs_value = rhs[j * k_ + n];

                let result = if compute_f16 {
                    (f16::from_f32(lhs_value) * f16::from_f32(rhs_value)).to_f32()
                } else {
                    lhs_value * rhs_value
                };

                out[i * n + j] += result;
            }
        }
    }
    out
}

pub fn test_matmul_cmma_1<R: Runtime>(device: &R::Device) {
    let m = 64;
    let k = 64;
    let n = 64;

    let factor = 100.;
    let tensor_1 = range_tensor_with_factor::<R>(m, k, factor, device);
    let tensor_2 = range_tensor_with_factor::<R>(k, n, factor, device);
    let out = Tensor {
        handle: create_empty::<R>(m, n, device),
        shape: vec![m, n],
        strides: vec![n, 1],
        elem: PhantomData,
    };

    let expected = matmul_cpu(
        f32::from_bytes(&R::client(device).read(tensor_1.handle.clone().binding())),
        f32::from_bytes(&R::client(device).read(tensor_2.handle.clone().binding())),
        m,
        k,
        n,
        true,
    );

    let out = matmul_cmma::<R, F32>(tensor_1, tensor_2, out, device);

    assert_equals::<R>(out.handle, &expected, device);
}

// pub fn test_matmul_cmma_2<R: Runtime>(device: &R::Device) {
//     let m = 256;
//     let k = 256;
//     let n = 256;

//     let tensor_1 = range_tensor::<R>(m, k, device);
//     let tensor_2 = range_tensor::<R>(k, n, device);
//     let out = Tensor {
//         handle: create_empty::<R>(m, n, device),
//         shape: vec![m, n],
//         strides: vec![n, 1],
//         elem: PhantomData,
//     };

//     let out = matmul_cmma::<R, F32>(tensor_1, tensor_2, out, device);

//     let client = R::client(device);
//     println!("{:?}", f32::from_bytes(&client.read(out.handle.binding())));
//     assert!(false);
// }
