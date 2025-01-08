use std::fmt::Display;

use cubecl_core::{
    prelude::{Numeric, Runtime},
    CubeElement,
};

use super::test_utils::eye_cpu;
use crate::tensor::TensorHandle;

pub fn test_eye<R: Runtime, C: Numeric + CubeElement + Display>(device: &R::Device, dim: u32) {
    let client = R::client(device);

    let expected = eye_cpu::<C>(dim);

    let eye = TensorHandle::<R, C>::eye(&client, dim);

    let actual = client.read_one(eye.handle.binding());
    let actual = C::from_bytes(&actual);

    assert_eq!(&expected[..], actual, "eye matrices are not equal.");
}
