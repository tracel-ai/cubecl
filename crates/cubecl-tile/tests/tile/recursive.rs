//! Recursive (multi-level) tiling
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};
use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{prelude::*, zspace::shape};
use cubecl_tile::{Axis, Partitioning, Space, TileArg};

use super::references;
use crate::tile::uncut;

const M: Axis = Axis(0);
const N: Axis = Axis(1);

/// An 8×8 tile, two nested levels of 2×2 sub-tiles
#[test]
fn recursive_two_level_tiled_view() {
    let client = cubecl::test_device().client();
    let (m, n) = (8usize, 8usize);

    let space = Space::new(&[(M, m), (N, n)]);
    let input = TileInput::builder(&client, space.clone())
        .split(&[2, 2])
        .split(&[2, 2])
        .arange();
    // Untiled output: its buffer is the logical shape itself, so `output[i * n + j]`
    // is the value read at logical `(i, j)`.
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    // The copy kernel only reads/writes through the views: no partitioning, so the
    // spaces carry no partitioner.
    let launch = uncut(&client, &space, &space);
    copy_logical::launch(
        &client,
        launch.cube_count(),
        launch.cube_dim(),
        input.arg(&launch, 1),
        output.arg(&launch, 1),
        launch.partitioning_arg(),
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = references::nested_index(i, j, n, &[2, 2]) as f32;
        }
    }
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();

    assert_equals_approx(&got, &expected, 1e-6).enforce()
}

/// Copy every logical element of `input` into `output` through their views.
#[cube(launch)]
fn copy_logical<E: Numeric>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let mut output = output.tile(&space);
    let r = input.view::<E, Const<1>>();
    let mut w = output.view_mut::<E, Const<1>>();
    let shape = r.shape();
    let rows = shape[0];
    let cols = shape[1];
    for i in 0..rows {
        for j in 0..cols {
            let mut pos = CoordsDyn::new();
            pos.push(i);
            pos.push(j);
            w.write(pos.clone(), r.read(pos));
        }
    }
}
