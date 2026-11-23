//! A sum a plane holds in manual-mma fragments, scaled and added to a procedural bias where it is
//! held ([`DimensionSlices::scale_add`]), then drained into a window that lists its dimensions in
//! the other order, a one-wide one between them, against the host.
//!
//! [`DimensionSlices::scale_add`]: cubecl_tile::DimensionSlices::scale_add

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{client::Client, prelude::*, std::tensor::TensorHandle, zspace::Shape};
use cubecl_tile::{
    Dimension, Levels, Partitioning, Scratch, Semiring, Space, TileSpec, TileTensor,
    TileTensorLaunch,
    procedural::{Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
};

use super::{Form, implied};

const M: Dimension = Dimension(0);
const N: Dimension = Dimension(1);
const K: Dimension = Dimension(2);
/// A head, one wide.
const H: Dimension = Dimension(3);

const SCALE: f32 = 2.0;
const SLOPE: f32 = 0.25;

/// What a cell is added to: how far its row lies past its column.
#[derive(CubeType, Clone)]
struct Slope {
    slope: f32,
}

#[cube]
impl Recipe<f32> for Slope {
    fn evaluate(&self, coordinates: &RecipeCoords) -> f32 {
        let (m, n) = (coordinates.along(M), coordinates.along(N));
        self.slope * (f32::cast_from(m) - f32::cast_from(n))
    }
}

impl Reads for SlopeExpand {
    fn reads(&self, _scope: &cubecl::ir::Scope, dimension: Dimension) -> bool {
        dimension == M || dimension == N
    }
}

#[cube(launch)]
fn scaled_and_drained_across(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    scale: f32,
    slope: f32,
    space: Partitioning,
) {
    let (lhs, rhs, out) = (lhs.tile(&space), rhs.tile(&space), out.tile_mut(&space));
    let dimensions = comptime!(space.space().subspace(&[M, N]));
    let bias =
        Procedural::new::<f32, Slope>(dimensions.clone(), Slope { slope }).tile_in::<f32>(&space);
    let shape = Procedural::zero::<f32>(dimensions).tile_in::<f32>(&space);
    for cube in space {
        for plane in cube {
            let (lhs_p, rhs_p) = (lhs.at(&plane), rhs.at(&plane));
            let mut sum = shape.at(&plane).rescaled_accumulator::<f32>(
                &lhs_p,
                &rhs_p,
                Semiring::SumProd,
                Scratch::None,
            );
            sum.zero();
            for fragment in plane.walk().unrolled() {
                let mut cell = sum.at(&fragment);
                cell.mma(&lhs_p.at(&fragment), &rhs_p.at(&fragment));
            }
            sum.along(N).scale_add::<f32>(scale, &bias.at(&plane));
            sum.drained_into(&out.at(&plane));
        }
    }
}

/// Every square `f16` instruction summing in `f32` that `client` offers as the tile's own transport
/// runs it.
fn square_instructions(client: &Client) -> Vec<usize> {
    let (f16_ty, f32_ty) = (half::f16::elem_type_native(), f32::elem_type_native());
    let mut sizes: Vec<usize> = (client.properties().features.matmul.mma.iter())
        .filter(|config| {
            [config.a_type, config.b_type, config.cd_type] == [f16_ty, f16_ty, f32_ty]
                && config.m == config.n
                && config.n == config.k
        })
        .map(|config| config.m as usize)
        .collect();
    sizes.dedup();
    sizes
}

fn input(client: &Client, shape: [usize; 2], salt: usize) -> (TensorHandle, Vec<f32>) {
    // Multiples of 1/64 within ±1, which `f16` holds exactly.
    let values: Vec<f32> = (0..shape[0] * shape[1])
        .map(|i| ((i * 2654435761 + salt * 40503) % 128) as f32 / 64. - 1.)
        .collect();
    let handle = TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(half::f16::elem_type_native())
        .custom(values.clone())
        .generate_without_host_data();
    (handle, values)
}

/// A plane's grid of fragments, its rows the output's last dimension.
fn check(client: &Client, tile: usize) {
    let (m, n, k) = (2 * tile, 3 * tile, 2 * tile);
    let (f16_ty, f32_ty) = (half::f16::elem_type_native(), f32::elem_type_native());
    let ((lhs, lhs_host), (rhs, rhs_host)) = (input(client, [m, k], 1), input(client, [n, k], 2));
    let out = TestInput::builder(client.clone(), Shape::new([n, 1, m]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let launcher = implied(
        client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k), (H, 1)]),
            Levels::leaf(&[(M, tile), (N, tile), (K, tile), (H, 1)])
                .walk(&[(M, 2), (N, 3), (K, 2)])
                .planes(&[(M, 1)])
                .cubes(&[M])
                .build(),
        ),
        Form::Static,
    );
    let arg = |handle: &TensorHandle, dtype, dimensions: &[Dimension]| {
        TileTensorLaunch::new(
            handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(dimensions),
        )
    };
    scaled_and_drained_across::launch(
        client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&lhs, f16_ty, &[M, K]),
        arg(&rhs, f16_ty, &[N, K]),
        arg(&out, f32_ty, &[N, H, M]),
        SCALE,
        SLOPE,
        launcher.partitioning_arg(),
    );
    let out = HostData::from_tensor_handle(client, out, HostDataType::F32);
    for row in 0..m {
        for column in 0..n {
            let product: f32 = (0..k)
                .map(|d| lhs_host[row * k + d] * rhs_host[column * k + d])
                .sum();
            let want = SCALE * product + SLOPE * (row as f32 - column as f32);
            let got = out.get_f32(&[column, 0, row]);
            assert!(
                (got - want).abs() <= 1e-3 * (1. + want.abs()),
                "{tile}: out[{column}, 0, {row}] = {got}, expected {want}"
            );
        }
    }
}

#[test]
fn a_held_sum_is_scaled_and_drained_into_a_transposed_window() {
    let client: Client = cubecl::test_device().client();
    let sizes = square_instructions(&client);
    if sizes.is_empty() {
        let why = "no square f16 instruction summing in f32".to_string();
        ValidationResult::Skipped(why).enforce();
        return;
    }
    sizes.into_iter().for_each(|tile| check(&client, tile));
}
