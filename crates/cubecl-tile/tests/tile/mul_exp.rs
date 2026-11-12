//! A score weighted by `exp(bias)` cell by cell instead of softmaxed ([`AxisSlices::mul_exp`]),
//! in a unit's registers and in a plane's manual-mma fragments, against the host. The bias is a
//! decay `-RATE · (query - key)` where the key is no later than the query, the minimum value
//! where it is later: those cells weigh zero.
//!
//! [`AxisSlices::mul_exp`]: cubecl_tile::AxisSlices::mul_exp

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Axis, Instruction, Levels, MmaOperands, Partitioning, RegisterBlock, Semiring, Space,
    StageStorage, Stages, Tile, TileSpec, TileTensor, TileTensorLaunch,
    ops::matmul::MmaIo,
    procedural::{Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
};

use super::{Form, implied};

const Q: Axis = Axis(0);
const S: Axis = Axis(1);
const D: Axis = Axis(2);
const V: Axis = Axis(3);

const RATE: f32 = 0.05;

/// The log-weight of a cell: a decay with the distance from the key to the query, the minimum
/// value for a key past its query.
#[derive(CubeType, Clone)]
struct Decay {
    rate: f32,
}

#[cube]
impl Recipe<f32> for Decay {
    fn evaluate(&self, coordinates: &RecipeCoords) -> f32 {
        let (q, s) = (coordinates.along(Q), coordinates.along(S));
        let behind = f32::cast_from(q) - f32::cast_from(s);
        select(s <= q, -self.rate * behind, f32::min_value())
    }
}

impl Reads for DecayExpand {
    fn reads(&self, _scope: &cubecl::ir::Scope, axis: Axis) -> bool {
        axis == S || axis == Q
    }
}

/// Every unit owning `rows` queries: a step of keys at a time, the score weighted in its
/// registers and drained.
#[cube(launch)]
fn by_units(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    weights: &TileTensor<'_>,
    rate: f32,
    space: Partitioning,
    #[comptime] rows: usize,
    #[comptime] block: usize,
) {
    let (query, key, weights) = (query.tile(&space), key.tile(&space), weights.tile(&space));
    let bias =
        Procedural::new::<f32, Decay>(comptime!(space.space().subspace(&[Q, S])), Decay { rate })
            .tile_in::<f32>(&space);
    let instruction = comptime!(Instruction::Registers {
        config: RegisterBlock::new(rows * block)
    });
    for cube in space {
        for unit in cube {
            let (query_u, key_u) = (query.at(&unit), key.at(&unit));
            let (weights_u, bias_u) = (weights.at(&unit), bias.at(&unit));
            for step in unit.walk() {
                let bias_s = bias_u.at(&step);
                let (query_s, key_s) = (query_u.at(&step), key_u.at(&step));
                let mut score =
                    bias_s.accumulator::<f32>(&query_s, &key_s, instruction, Semiring::SUM_PROD);
                score.mma(&query_s, &key_s);
                score.along(S).mul_exp::<f32>(&bias_s);
                score.drained_into(&weights_u.at(&step));
            }
        }
    }
}

/// Every plane owning its queries, its scores in manual-mma fragments: a stage of keys at a time,
/// the score weighted in the registers that hold it, or, `windowed`, drained into the plane's
/// window first and weighted there, and drained.
#[cube(launch)]
fn by_plane_fragments(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    weights: &TileTensor<'_>,
    rate: f32,
    space: Partitioning,
    #[comptime] windowed: bool,
) {
    let instruction = comptime!(Instruction::Mma {
        io: MmaIo::default(),
        operands: MmaOperands::Plain,
    });
    let (query, key, weights) = (query.tile(&space), key.tile(&space), weights.tile(&space));
    let bias =
        Procedural::new::<f32, Decay>(comptime!(space.space().subspace(&[Q, S])), Decay { rate })
            .tile_in::<f32>(&space);
    for cube in space {
        for plane in cube {
            let walk = plane.walk();
            let mut query_s = query
                .at(&plane)
                .stage_for(&walk, StageStorage::Strided, None);
            query_s.copy_from(&query.at(&plane));
            sync_plane();
            let scores =
                Tile::stage_accumulator::<f32>(&walk, comptime!(vec![Q, S]), &query_s, instruction);
            let window = Tile::scratch::<f32>(&walk, comptime!(vec![Q, S]), StageStorage::Strided);
            let (bias_p, weights_p) = (bias.at(&plane), weights.at(&plane));
            let mut stages =
                Stages::smem_single(&walk, &key.at(&plane), StageStorage::Strided, 1usize, None);
            stages.pipelined(walk, |slot, stage| {
                let bias_s = bias_p.at(stage);
                slot.consume(|key_s| {
                    let mut score = scores.clone();
                    score.reset();
                    for fragment in stage.walk().routed(V, 0).unrolled() {
                        let mut cell = score.at(&fragment);
                        cell.mma(&query_s.at(&fragment), &key_s.at(&fragment));
                    }
                    if windowed {
                        score.drained_into(&window);
                        sync_plane();
                        window.along(S).mul_exp::<f32>(&bias_s);
                        sync_plane();
                        weights_p.at(stage).copy_from(&window);
                    } else {
                        score.along(S).mul_exp::<f32>(&bias_s);
                        score.drained_into(&weights_p.at(stage));
                    }
                });
            });
        }
    }
}

/// The problem's shape. The last key lies past every query.
struct Case {
    queries: usize,
    keys: usize,
    head: usize,
}

impl Case {
    /// The inputs on `client`, with their host copies.
    fn inputs(
        &self,
        client: &Client,
        dtype: cubecl::ir::ElemType,
    ) -> (Vec<f64>, [cubecl::std::tensor::TensorHandle; 2]) {
        // Multiples of 1/512 within ±2, which `f16` holds exactly.
        let wobble =
            |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
        let keys = self.keys;
        let input = |shape: [usize; 2], salt: usize| {
            let mut values: Vec<f32> = (0..shape[0] * shape[1]).map(|i| wobble(i, salt)).collect();
            // The last key is infinite and past every query: its masked scores are not finite.
            if salt == 2 {
                values[(keys - 1) * shape[1]..].fill(f32::INFINITY);
            }
            let handle = TestInput::builder(client.clone(), Shape::new(shape))
                .dtype(dtype)
                .custom(values.clone())
                .generate_without_host_data();
            (handle, values)
        };
        let ((query_handle, query), (key_handle, key)) = (
            input([self.queries, self.head], 1),
            input([self.keys, self.head], 2),
        );
        let expected = (0..self.queries * self.keys)
            .map(|at| {
                let (i, j) = (at / self.keys, at % self.keys);
                match j <= i {
                    true => {
                        let score: f64 = (0..self.head)
                            .map(|d| {
                                query[i * self.head + d] as f64 * key[j * self.head + d] as f64
                            })
                            .sum();
                        score * (-(RATE as f64) * (i - j) as f64).exp()
                    }
                    false => 0.,
                }
            })
            .collect();
        (expected, [query_handle, key_handle])
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
    axes: &[Axis],
) -> TileTensorLaunch {
    TileTensorLaunch::new(
        handle.clone().binding().into_tensor_arg(),
        dtype,
        1,
        TileSpec::direct(axes),
    )
}

fn assert_weights(
    client: &Client,
    weights: cubecl::std::tensor::TensorHandle,
    case: &Case,
    expected: &[f64],
    tolerance: f64,
) {
    let weights = HostData::from_tensor_handle(client, weights, HostDataType::F32);
    for i in 0..case.queries {
        for j in 0..case.keys {
            let (got, want) = (weights.get_f32(&[i, j]) as f64, expected[i * case.keys + j]);
            assert!(
                (got - want).abs() <= tolerance * (1. + want.abs()),
                "weights[{i}, {j}] = {got}, expected {want}"
            );
        }
    }
}

/// Units weigh the scores they hold in registers by the decay: each cell is its score times
/// `exp(bias)`, and zero where the bias is masked.
#[test]
fn units_weigh_their_scores_by_the_exponential_of_a_bias() {
    let client: Client = cubecl::test_device().client();
    let (rows, block, units) = (2, 4, 4);
    let case = Case {
        queries: 16,
        keys: 24,
        head: 8,
    };
    let f32_ty = f32::elem_type_native();
    let (expected, [query, key]) = case.inputs(&client, f32_ty);
    let weights = zeros(&client, [case.queries, case.keys]);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, case.queries), (S, case.keys), (D, case.head)]),
            Levels::leaf(&[(Q, rows), (S, block), (D, case.head)])
                .walk_every(&[S])
                .units_distributed(Q, units)
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    by_units::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&query, f32_ty, &[Q, D]),
        arg(&key, f32_ty, &[S, D]),
        arg(&weights, f32_ty, &[Q, S]),
        RATE,
        launcher.partitioning_arg(),
        rows,
        block,
    );
    assert_weights(&client, weights, &case, &expected, 1e-4);
}

/// Every `[m, n, k]` instruction `client` offers on `f16` summing in `f32`.
fn f16_mma_shapes(client: &Client) -> Vec<[usize; 3]> {
    let (f16_ty, f32_ty) = (half::f16::elem_type_native(), f32::elem_type_native());
    let mut shapes: Vec<[usize; 3]> = client
        .properties()
        .features
        .matmul
        .mma
        .iter()
        .filter(|cfg| cfg.a_type == f16_ty && cfg.b_type == f16_ty && cfg.cd_type == f32_ty)
        .map(|cfg| [cfg.m as usize, cfg.n as usize, cfg.k as usize])
        .collect();
    shapes.sort();
    shapes.dedup();
    shapes
}

/// A plane weighs the scores it holds in manual-mma fragments by the decay, in the registers that
/// hold them, across every fragment of its grid, for every `f16` instruction the device offers.
#[test]
fn plane_fragments_weigh_their_scores_by_the_exponential_of_a_bias() {
    for_each_f16_shape(|client, shape| plane_weighs(client, shape, false));
}

/// A plane weighs the scores it drained into its window, shared memory, the same way.
#[test]
fn a_plane_window_weighs_its_scores_by_the_exponential_of_a_bias() {
    for_each_f16_shape(|client, shape| plane_weighs(client, shape, true));
}

fn for_each_f16_shape(run: impl Fn(&Client, [usize; 3])) {
    let client: Client = cubecl::test_device().client();
    let shapes = f16_mma_shapes(&client);
    if shapes.is_empty() {
        ValidationResult::Skipped("no f16 instruction summing in f32".to_string()).enforce();
        return;
    }
    for shape in shapes {
        run(&client, shape);
    }
}

/// One plane owning two `m` fragments of queries, walking the keys in stages of two `n`
/// fragments, over a head of two `k`.
fn plane_weighs(client: &Client, [m, n, k]: [usize; 3], windowed: bool) {
    let case = Case {
        queries: 2 * m,
        keys: 8 * n,
        head: 2 * k,
    };
    assert!(
        case.keys > case.queries,
        "the last key must lie past every query"
    );
    let f16_ty = half::f16::elem_type_native();
    let (expected, [query, key]) = case.inputs(client, f16_ty);
    let weights = zeros(client, [case.queries, case.keys]);
    let launcher = implied(
        client,
        Partitioning::new(
            Space::new(&[
                (Q, case.queries),
                (S, case.keys),
                (D, case.head),
                (V, case.head),
            ]),
            Levels::leaf(&[(Q, m), (S, n), (D, k), (V, k)])
                .walk(&[(Q, 2), (S, 2), (D, 2), (V, 2)])
                .walk_every(&[S])
                .planes(&[(Q, 1)])
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    by_plane_fragments::launch(
        client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&query, f16_ty, &[Q, D]),
        arg(&key, f16_ty, &[S, D]),
        arg(&weights, f32::elem_type_native(), &[Q, S]),
        RATE,
        launcher.partitioning_arg(),
        windowed,
    );
    // f16 operands into f32 sums: a few ulps of a value near one.
    assert_weights(client, weights, &case, &expected, 2e-3);
}
