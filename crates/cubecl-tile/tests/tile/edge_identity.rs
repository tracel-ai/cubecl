//! A read past the problem's edge of an operand bound with [`Boundary::Identity`] returns the
//! identity of its monoid, where [`Boundary::Zero`] returns zero.

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::shape};

use super::{Form, implied};
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};

const M: Dimension = Dimension(0);
const K: Dimension = Dimension(1);

/// Every line of each region of `input`, past the edge included, copied into `out`, which is wide
/// enough to hold them.
#[cube(launch)]
fn copy_past_edge_kernel(
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

/// `m` rows of `k` negative values read in boxes of `tk` along the row, `k` no multiple of `tk`,
/// bound with `boundary`: the output, `tk`-padded rows.
fn copied(m: usize, k: usize, tk: usize, boundary: Boundary) -> (Vec<f32>, HostData) {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let padded = k.div_ceil(tk) * tk;
    let data: Vec<f32> = (0..m * k).map(|i| -1.0 - i as f32).collect();
    let (input, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(f32_ty)
        .custom(data.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, padded])
        .dtype(f32_ty)
        .custom(vec![7.0; m * padded])
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, 1), (K, tk)]).walk_every(&[M, K]).build(),
        ),
        Form::Static,
    );
    copy_past_edge_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileTensorLaunch::new(
            input.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[M, K]).boundary(BoundaryPolicy::Every(boundary)),
        ),
        TileTensorLaunch::new(
            out.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[M, K]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );
    (data, HostData::from_tensor_handle(&client, out, HostDataType::F32))
}

/// Every row's cells: its values, then `fill` past its edge.
fn expect(got: &HostData, data: &[f32], m: usize, k: usize, padded: usize, fill: f32) {
    for row in 0..m {
        for col in 0..padded {
            let want = match col < k {
                true => data[row * k + col],
                false => fill,
            };
            assert_eq!(got.get_f32(&[row, col]), want, "row {row}, col {col}");
        }
    }
}

#[test]
fn a_read_past_the_edge_returns_the_max_identity() {
    let (m, k, tk) = (2, 6, 4);
    let (data, got) = copied(m, k, tk, Boundary::Identity(Monoid::Max));
    expect(&got, &data, m, k, 8, f32::NEG_INFINITY);
}

#[test]
fn a_read_past_the_edge_returns_the_min_identity() {
    let (m, k, tk) = (2, 6, 4);
    let (data, got) = copied(m, k, tk, Boundary::Identity(Monoid::Min));
    expect(&got, &data, m, k, 8, f32::INFINITY);
}

#[test]
fn a_read_past_the_edge_returns_zero_under_zero() {
    let (m, k, tk) = (2, 6, 4);
    let (data, got) = copied(m, k, tk, Boundary::Zero);
    expect(&got, &data, m, k, 8, 0.0);
}
