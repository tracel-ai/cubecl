use std::fmt::Display;

use cubecl_core::{
    prelude::{Numeric, Runtime},
    CubeElement,
};

use super::test_utils::identity_cpu;
use crate::tensor::{self, TensorHandle};

pub fn test_identity<R: Runtime, C: Numeric + CubeElement + Display>(
    device: &R::Device,
    dim: usize,
) {
    let client = R::client(device);

    let expected = identity_cpu::<C>(dim);

    let identity = TensorHandle::<R, C>::empty(&client, [dim, dim].to_vec());
    tensor::identity::launch(&client, &identity);

    let actual = client.read_one(identity.handle.clone().binding());
    let actual = C::from_bytes(&actual);

    assert_eq!(&expected[..], actual, "identity matrices are not equal.");
}
