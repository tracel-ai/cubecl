//! A plane's window of a global buffer read in loads of four values, its slices stepped by
//! [`DimensionSlices`] against the host.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_tile::procedural::{MemoryValues, Procedural};
use cubecl_tile::*;

const ROW: Dimension = Dimension(0);
const COL: Dimension = Dimension(1);
const ROWS: usize = 4;
/// Longer than a plane's loads, so each unit steps more than one load of a slice.
const COLS: usize = 256;
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

/// Every plane of the launch owning two rows:
/// `gradient = probabilities · (gradient − delta[row])` in place.
#[cube(launch)]
fn softmax_gradient_window(
    gradient: &mut TileTensor<'_>,
    probabilities: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    space: Partitioning,
) {
    let gradient = gradient.tile_mut(&space);
    let probabilities = probabilities.tile(&space);
    let delta = delta.tile(&space);
    for cube in space {
        for plane in cube {
            let rows = gradient.at(&plane);
            rows.along(COL)
                .softmax_gradient::<f32>(&probabilities.at(&plane).along(COL), &delta.at(&plane));
        }
    }
}

fn value(row: usize, col: usize) -> f32 {
    (row * COLS + col) as f32 / 8. - 3.
}

fn bias(col: usize) -> f32 {
    col as f32 * 0.25 + 1.
}

fn probability(row: usize, col: usize) -> f32 {
    ((row + col) % 7) as f32 / 8.
}

fn delta(row: usize) -> f32 {
    row as f32 * 0.5 - 1.
}

fn input<const D: usize>(client: &Client, values: Vec<f32>, dims: [usize; D]) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new(dims))
        .dtype(f32::elem_type_native())
        .custom(values)
        .generate_without_host_data()
}

fn grid(cell: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    (0..ROWS * COLS).map(|i| cell(i / COLS, i % COLS)).collect()
}

/// A plane per two rows, a cube per plane.
fn launcher(client: &Client) -> Launcher {
    implied(
        client,
        Partitioning::new(
            Space::new(&[(ROW, ROWS), (COL, COLS)]),
            Levels::leaf(&[(ROW, 2), (COL, COLS)])
                .planes(&[(ROW, 1)])
                .cubes(&[ROW])
                .build(),
        ),
        Form::Static,
    )
}

/// `handle` bound `width` values a load, as `spec` says.
fn arg(handle: &TensorHandle, width: usize, spec: TileSpec) -> TileTensorLaunch {
    TileTensorLaunch::new(
        handle.clone().binding().into_tensor_arg(),
        f32::elem_type_native(),
        width,
        spec,
    )
}

fn assert_grid(client: &Client, got: TensorHandle, want: impl Fn(usize, usize) -> f32) {
    let got = HostData::from_tensor_handle(client, got, HostDataType::F32);
    for row in 0..ROWS {
        for col in 0..COLS {
            let (have, want) = (got.get_f32(&[row, col]), want(row, col));
            assert!(
                (have - want).abs() < 1e-5,
                "at ({row}, {col}): got {have}, want {want}"
            );
        }
    }
}

/// A window read four values a load adds the bias of its column to every cell, not of the load.
#[test]
fn a_window_in_loads_of_four_values_scales_and_adds_a_bias() {
    let client: Client = cubecl::test_device().client();
    let window = input(&client, grid(value), [ROWS, COLS]);
    let bias_values = input(&client, (0..COLS).map(bias).collect(), [COLS]);
    let launcher = launcher(&client);
    scale_add_window::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&window, 4, TileSpec::direct(&[ROW, COL])),
        arg(&bias_values, 1, TileSpec::direct(&[COL])),
        SCALE,
        launcher.partitioning_arg(),
    );
    assert_grid(&client, window, |row, col| {
        value(row, col) * SCALE + bias(col)
    });
}

/// Gradients and probabilities both read four values a load take the softmax gradient of every
/// cell.
#[test]
fn windows_in_loads_of_four_values_take_the_softmax_gradient() {
    let client: Client = cubecl::test_device().client();
    let gradient = input(&client, grid(value), [ROWS, COLS]);
    let probabilities = input(&client, grid(probability), [ROWS, COLS]);
    let deltas = input(&client, (0..ROWS).map(delta).collect(), [ROWS]);
    let launcher = launcher(&client);
    softmax_gradient_window::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&gradient, 4, TileSpec::direct(&[ROW, COL])),
        arg(&probabilities, 4, TileSpec::direct(&[ROW, COL])),
        arg(&deltas, 1, TileSpec::direct(&[ROW])),
        launcher.partitioning_arg(),
    );
    assert_grid(&client, gradient, |row, col| {
        probability(row, col) * (value(row, col) - delta(row))
    });
}
