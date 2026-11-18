//! A dimension spanning several axes of a plain buffer: `[a, k, c]` read as `B = a · c` rows of
//! `k`, `B` spanning the axes `a` and `c` around `k`.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::shape};

use super::{Form, implied};
use cubecl_tile::*;

const B: Dimension = Dimension(0);
const K: Dimension = Dimension(1);

/// Every region of `input` copied into `out`, line by line.
#[cube(launch)]
fn copy_kernel(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let x = input.tile(&space);
    let o = out.tile(&space);
    for region in space.over(&level) {
        let window = x.at(&region);
        let lines = window.flat::<f32, Const<1>>();
        let mut written = o.at(&region);
        let mut cells = written.flat_mut::<f32, Const<1>>();
        for i in 0..lines.shape() {
            cells.write(i, lines.read(i));
        }
    }
}

/// `[a, k, c]` copied into `[a · c, k]` through a `B` that spans `a` and `c`, in boxes of `rows`
/// rows: `out[b, j]` is `input[b / c, j, b % c]`.
fn folded_copy(a: usize, k: usize, c: usize, rows: usize) {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let data: Vec<f32> = (0..a * k * c).map(|i| i as f32).collect();
    let (input, _) = TestInput::builder(client.clone(), shape![a, k, c])
        .dtype(f32_ty)
        .custom(data.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![a * c, k])
        .dtype(f32_ty)
        .custom(vec![-1.0; a * c * k])
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, a * c), (K, k)]),
            Levels::leaf(&[(B, rows), (K, k)]).walk_every(&[B, K]).build(),
        ),
        Form::Static,
    );
    copy_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileTensorLaunch::new(
            input.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[B, K, B]),
        ),
        TileTensorLaunch::new(
            out.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[B, K]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for b in 0..a * c {
        for j in 0..k {
            let want = data[(b / c) * k * c + j * c + b % c];
            assert_eq!(got.get_f32(&[b, j]), want, "row {b}, value {j}");
        }
    }
}

#[test]
fn rows_spanning_two_axes_around_the_reduced_one_read_in_place() {
    folded_copy(3, 5, 8, 1);
}

#[test]
fn a_box_of_rows_inside_the_inner_axis_reads_in_place() {
    folded_copy(3, 5, 8, 4);
}

/// Every region of `input` copied into `out`, four values a line.
#[cube(launch)]
fn copy_lines_kernel(
    input: &TileTensor<'_>,
    out: &TileTensor<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let x = input.tile(&space);
    let o = out.tile(&space);
    for region in space.over(&level) {
        let window = x.at(&region);
        let lines = window.flat::<f32, Const<4>>();
        let mut written = o.at(&region);
        let mut cells = written.flat_mut::<f32, Const<4>>();
        for i in 0..lines.shape() {
            cells.write(i, lines.read(i));
        }
    }
}

/// `[a, k, c]` copied into a buffer of the same shape, both read through a `B` that spans `a`
/// and `c` in lines of four along `c`, a box of four rows at a time: the copy is exact.
#[test]
fn lines_of_rows_inside_the_inner_axis_read_in_place() {
    let (a, k, c) = (3, 5, 8);
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let data: Vec<f32> = (0..a * k * c).map(|i| i as f32).collect();
    let (input, _) = TestInput::builder(client.clone(), shape![a, k, c])
        .dtype(f32_ty)
        .custom(data.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![a, k, c])
        .dtype(f32_ty)
        .custom(vec![-1.0; a * k * c])
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(K, k), (B, a * c)]),
            Levels::leaf(&[(K, 1), (B, 4)]).walk_every(&[K, B]).build(),
        ),
        Form::Static,
    );
    copy_lines_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileTensorLaunch::new(
            input.binding().into_tensor_arg(),
            f32_ty,
            4,
            TileSpec::direct(&[B, K, B]),
        ),
        TileTensorLaunch::new(
            out.clone().binding().into_tensor_arg(),
            f32_ty,
            4,
            TileSpec::direct(&[B, K, B]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for i in 0..a {
        for j in 0..k {
            for l in 0..c {
                let want = data[i * k * c + j * c + l];
                assert_eq!(got.get_f32(&[i, j, l]), want, "at [{i}, {j}, {l}]");
            }
        }
    }
}
