use std::fmt::Display;

use cubecl_core::{CubeElement, Runtime, prelude::Float};
use cubecl_std::tensor::TensorHandle;

use crate::tests::test_utils::{assert_equals_approx, tensorhandler_from_data, transpose_matrix};

pub fn test_qr_cgr<R: Runtime, F: Float + CubeElement + Display>(device: &R::Device, dim: u32) {
    let client = R::client(device);
    let dim_usize = dim as usize;

    let shape = vec![dim as usize, dim as usize];
    let num_elements = shape.iter().product();
    let mut data = vec![F::from_int(1); num_elements];
    let mut pos = dim_usize - 1;
    for _i in 0..dim {
        data[pos] = F::from_int(2);
        pos += dim_usize - 1;
    }

    let a = tensorhandler_from_data::<R, F>(&client, shape.clone(), &data);

    /*let bytes = client.read_one_tensor(a.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("A Output => {output:?}"); */

    let (mut q_t, r) =
        match crate::launch::<R, F>(&crate::Strategy::CommonGivensRotations, &client, &a) {
            Ok((q_t, r)) => (q_t, r),
            Err(_) => (
                TensorHandle::empty(&client, shape.clone()),
                TensorHandle::empty(&client, shape.clone()),
            ),
        };

    /*let bytes = client.read_one_tensor(q.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("Q Output => {output:?}");

    let bytes = client.read_one_tensor(r.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("R Output => {output:?}");*/

    let q = transpose_matrix(&client, &mut q_t);

    /*let bytes = client.read_one_tensor(q_t.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("Q Transposed Output => {output:?}");*/

    let out: TensorHandle<R, F> = TensorHandle::empty(&client, shape.clone());
    cubecl_matmul::kernels::naive::launch::<R, F, F>(
        &client,
        cubecl_matmul::MatmulInputHandle::Normal(q),
        cubecl_matmul::MatmulInputHandle::Normal(r),
        &out.as_ref(),
    )
    .unwrap();

    /*let bytes = client.read_one_tensor(out.as_copy_descriptor());
    let output = F::from_bytes(&bytes);
    println!("Result Output => {output:?}");*/

    if let Err(e) =
        assert_equals_approx::<R, F>(&client, out.handle, &out.shape, &out.strides, &data, 10e-3)
    {
        panic!("{}", e);
    }
}
