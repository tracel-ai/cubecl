//! A per-column bias read from a buffer, added to a scaled cmma accumulator drained into each
//! plane's window ([`DimensionSlices::scale_add`]), against the host.
//!
//! [`DimensionSlices::scale_add`]: cubecl_tile::DimensionSlices::scale_add

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Dimension, Levels, Partitioning, Semiring, Space, StageStorage, Stages, Tile, TileSpec,
    TileTensor, TileTensorLaunch,
    procedural::{MemoryValues, Procedural},
};

use super::{Form, implied};

const Q: Dimension = Dimension(0);
const S: Dimension = Dimension(1);
const D: Dimension = Dimension(2);

const SCALE: f32 = 0.5;

/// The bias as a procedural tile over `[Q, S]` reading the buffer along `S`, so it broadcasts
/// over the rows.
#[cube]
fn column_bias(bias: &TileTensor<'_>, space: &Partitioning) -> Tile {
    let values: MemoryValues = bias.tile(space).values();
    Procedural::new::<f32, MemoryValues>(comptime!(space.space().subspace(&[Q, S])), values)
        .tile_in::<f32>(space)
}

/// A cmma matmul over a K walk, each plane's accumulator drained into a window of shared memory
/// the plane holds, scaled and biased there, and copied out: `along` reads a window only from the
/// plane holding it, so a cube-level accumulator is drained plane by plane.
#[cube(launch)]
fn cmma_window(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    bias: &TileTensor<'_>,
    out: &mut TileTensor<'_>,
    scale: f32,
    space: Partitioning,
    #[comptime] planes: cubecl_tile::Level,
    #[comptime] steps: cubecl_tile::Level,
) {
    let (query, key, out) = (query.tile(&space), key.tile(&space), out.tile_mut(&space));
    let bias = column_bias(bias, &space);
    for cube in space {
        for plane in cube.clone() {
            let (query, key) = (query.at(&plane), key.at(&plane));
            let mut out = out.at(&plane);
            let acc = out.accumulator::<f32>(&query, &key, Semiring::SumProd);
            let walk = plane.over(&steps);
            let mut stages = Stages::smem(&walk, &query, &key, StageStorage::Strided, 1usize, None);
            stages.pipelined(walk, |slot, region| {
                let mut acc_r = acc.at(region);
                slot.consume(|query_s, key_s| {
                    acc_r.mma(query_s, key_s);
                });
            });
            let window = Tile::scratch::<f32>(
                &cube.over(&planes),
                comptime!(vec![Q, S]),
                StageStorage::Strided,
            );
            acc.drained_into(&window);
            sync_plane();
            window.along(S).scale_add::<f32>(scale, &bias.at(&plane));
            sync_plane();
            out.copy_from(&window);
        }
    }
}

/// The problem's shape.
struct Case {
    queries: usize,
    keys: usize,
    head: usize,
}

impl Case {
    /// The inputs on `client`, with the expected `acc · SCALE + bias[column]`.
    fn inputs(
        &self,
        client: &Client,
        dtype: cubecl::ir::ElemType,
    ) -> (Vec<f64>, [cubecl::std::tensor::TensorHandle; 3]) {
        // Multiples of 1/512 within ±2, which `f16` holds exactly.
        let wobble =
            |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
        let input = |shape: &[usize], salt: usize| {
            let values: Vec<f32> = (0..shape.iter().product())
                .map(|i| wobble(i, salt))
                .collect();
            let handle = TestInput::builder(client.clone(), Shape::from(shape.to_vec()))
                .dtype(dtype)
                .custom(values.clone())
                .generate_without_host_data();
            (handle, values)
        };
        let ((query_handle, query), (key_handle, key), (bias_handle, bias)) = (
            input(&[self.queries, self.head], 1),
            input(&[self.keys, self.head], 2),
            input(&[self.keys], 3),
        );
        let expected = (0..self.queries * self.keys)
            .map(|at| {
                let (i, j) = (at / self.keys, at % self.keys);
                let score: f64 = (0..self.head)
                    .map(|d| query[i * self.head + d] as f64 * key[j * self.head + d] as f64)
                    .sum();
                score * SCALE as f64 + bias[j] as f64
            })
            .collect();
        (expected, [query_handle, key_handle, bias_handle])
    }
}

fn zeros(client: &Client, shape: [usize; 2]) -> cubecl::std::tensor::TensorHandle {
    TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data()
}

fn arg(
    handle: &cubecl::std::tensor::TensorHandle,
    dtype: cubecl::ir::ElemType,
    dimensions: &[Dimension],
) -> TileTensorLaunch {
    TileTensorLaunch::new(
        handle.clone().binding().into_tensor_arg(),
        dtype,
        1,
        TileSpec::direct(dimensions),
    )
}

fn assert_out(
    client: &Client,
    out: cubecl::std::tensor::TensorHandle,
    case: &Case,
    expected: &[f64],
    tolerance: f64,
) {
    let out = HostData::from_tensor_handle(client, out, HostDataType::F32);
    for i in 0..case.queries {
        for j in 0..case.keys {
            let (got, want) = (out.get_f32(&[i, j]) as f64, expected[i * case.keys + j]);
            assert!(
                (got - want).abs() <= tolerance * (1. + want.abs()),
                "out[{i}, {j}] = {got}, expected {want}"
            );
        }
    }
}

/// Every `[m, n, k]` cmma fragment `client` offers on `f16` summing in `f32`.
fn f16_cmma_shapes(client: &Client) -> Vec<[usize; 3]> {
    let (f16_ty, f32_ty) = (half::f16::elem_type_native(), f32::elem_type_native());
    let mut shapes: Vec<[usize; 3]> = client
        .properties()
        .features
        .matmul
        .cmma
        .iter()
        .filter(|cfg| cfg.a_type == f16_ty && cfg.b_type == f16_ty && cfg.cd_type == f32_ty)
        .map(|cfg| [cfg.m as usize, cfg.n as usize, cfg.k as usize])
        .collect();
    shapes.sort();
    shapes.dedup();
    shapes
}

/// A cmma accumulator of two `m` fragments by one `n`, drained into a window and biased there,
/// over a K walk of two `k` stages. The fragment is `16x16x16` where the device offers it, else
/// the first `f16` shape it lists.
#[test]
fn a_cmma_accumulator_drained_into_a_window_is_scaled_and_biased() {
    let client: Client = cubecl::test_device().client();
    let shapes = f16_cmma_shapes(&client);
    let Some(&[m, n, k]) = shapes
        .iter()
        .find(|s| **s == [16, 16, 16])
        .or(shapes.first())
    else {
        ValidationResult::Skipped("no f16 cmma fragment summing in f32".to_string()).enforce();
        return;
    };
    let case = Case {
        queries: 2 * m,
        keys: n,
        head: 2 * k,
    };
    let f16_ty = half::f16::elem_type_native();
    let (expected, [query, key, bias]) = case.inputs(&client, f16_ty);
    let out = zeros(&client, [case.queries, case.keys]);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, case.queries), (S, case.keys), (D, case.head)]),
            Levels::leaf(&[(Q, m), (S, n), (D, k)])
                .walk(&[(Q, 2), (S, 1)])
                .walk(&[(D, 1)])
                .walk_every(&[D])
                .planes(&[(Q, 1), (S, 1)])
                .cubes(&[Q, S])
                .build(),
        ),
        Form::Static,
    );
    let f32_ty = f32::elem_type_native();
    cmma_window::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&query, f16_ty, &[Q, D]),
        arg(&key, f16_ty, &[S, D]),
        arg(&bias, f16_ty, &[S]),
        arg(&out, f32_ty, &[Q, S]),
        SCALE,
        launcher.partitioning_arg(),
        launcher.partitioning().level(1),
        launcher.partitioning().level(2),
    );
    assert_out(&client, out, &case, &expected, 2e-3);
}
