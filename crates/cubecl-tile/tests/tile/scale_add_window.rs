//! A plane's window of a global buffer scaled and added a per-column bias slice by slice
//! ([`DimensionSlices::scale_add`]), the buffer bound in loads of one value and of four along its
//! innermost dimension, against the host.
//!
//! [`DimensionSlices::scale_add`]: cubecl_tile::DimensionSlices::scale_add

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_tile::procedural::{MemoryValues, Procedural};
use cubecl_tile::*;

const ROW: Dimension = Dimension(0);
const COL: Dimension = Dimension(1);
const ROWS: usize = 4;
const COLS: usize = 16;
const SCALE: f32 = 0.5;

/// Every plane of the launch owning two rows: `window = window · scale + bias[col]` in place.
#[cube(launch)]
fn scale_add_window(
    window: &mut TileTensor<'_>,
    bias: &TileTensor<'_>,
    scale: f32,
    space: Partitioning,
) {
    let window = window.tile_mut(&space);
    let bias = Procedural::new::<f32, MemoryValues>(
        comptime!(space.space().subspace(&[COL])),
        bias.tile(&space).values(),
    )
    .tile_in::<f32>(&space);
    for cube in space {
        for plane in cube {
            let rows = window.at(&plane);
            rows.along(COL).scale_add::<f32>(scale, &bias.at(&plane));
        }
    }
}

fn value(row: usize, col: usize) -> f32 {
    (row * COLS + col) as f32 / 8. - 3.
}

fn bias(col: usize) -> f32 {
    col as f32 * 0.25 + 1.
}

fn input<const D: usize>(client: &Client, values: Vec<f32>, dims: [usize; D]) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new(dims))
        .dtype(f32::elem_type_native())
        .custom(values)
        .generate_without_host_data()
}

/// Scale and add over a window bound `width` values a load, and check every cell.
fn check(width: usize) {
    let client: Client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let cells = (0..ROWS * COLS)
        .map(|i| value(i / COLS, i % COLS))
        .collect();
    let window = input(&client, cells, [ROWS, COLS]);
    let bias_values = input(&client, (0..COLS).map(bias).collect(), [COLS]);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, 2), (COL, COLS)])
                .planes(&[(ROW, 1)])
                .cubes(&[ROW])
                .build(),
        ),
        Form::Static,
    );
    scale_add_window::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileTensorLaunch::new(
            window.clone().binding().into_tensor_arg(),
            f32_ty,
            width,
            TileSpec::direct(&[ROW, COL]),
        ),
        TileTensorLaunch::new(
            bias_values.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[COL]),
        ),
        SCALE,
        launcher.partitioning_arg(),
    );
    let got = HostData::from_tensor_handle(&client, window, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..COLS {
            let want = value(row, col) * SCALE + bias(col);
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-5,
                "width {width}, at ({row}, {col}): got {have}, want {want}"
            );
        }
    }
}

/// A window read one value a load adds the bias of its column to every cell.
#[test]
fn a_window_in_loads_of_one_value_scales_and_adds_a_bias() {
    check(1);
}

/// A window read four values a load adds the bias of its column to every cell, not of the load.
#[test]
fn a_window_in_loads_of_four_values_scales_and_adds_a_bias() {
    check(4);
}
