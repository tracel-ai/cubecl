//! The softmax's backward pass on tiles a unit holds in registers: its probabilities read back
//! from a saved logsumexp ([`AxisSlices::probabilities`]), its gradient against the scores
//! ([`AxisSlices::softmax_gradient`]), the dot product that gradient subtracts ([`Tile::dot`]),
//! and the logsumexp an online softmax leaves behind
//! ([`OnlineSoftmax::store_logsumexp`](cubecl_tile::ops::softmax::OnlineSoftmax::store_logsumexp)).
//!
//! The logsumexp and the dot product are tiles over the queries, which the scores broadcast
//! along the axis they omit. A unit that owns queries holds its scores `[q, s]` and reads one
//! value a slice; a unit that owns keys holds them `[s, q]`, as attention's backward pass does to
//! sum the keys' gradients, and reads one value a position along each slice.
//!
//! [`AxisSlices::probabilities`]: cubecl_tile::AxisSlices::probabilities
//! [`AxisSlices::softmax_gradient`]: cubecl_tile::AxisSlices::softmax_gradient
//! [`Tile::dot`]: cubecl_tile::Tile::dot

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Axis, Instruction, Levels, Partitioning, RegisterBlock, Semiring, Space, TileArg,
    TileArgLaunch, TileSpec,
    ops::softmax::OnlineSoftmax,
    procedural::{Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
};

use super::{Form, implied};

const Q: Axis = Axis(0);
const S: Axis = Axis(1);
const D: Axis = Axis(2);
const V: Axis = Axis(3);

/// The bias a score is summed with: zero at a key its query sees, the minimum value past its own
/// position, aligned at the bottom right.
#[derive(CubeType, Clone)]
struct Causal {
    keys: u32,
    queries: u32,
}

#[cube]
impl Recipe<f32> for Causal {
    fn evaluate(&self, coordinates: &RecipeCoords) -> f32 {
        let seen = coordinates.along(S) + self.queries <= coordinates.along(Q) + self.keys;
        select(seen, 0.0, f32::min_value())
    }
}

impl Reads for CausalExpand {
    fn reads(&self, _scope: &cubecl::ir::Scope, axis: Axis) -> bool {
        axis == S || axis == Q
    }
}

/// Every unit owning `rows` queries: their dot product of the output and its gradient, then, a
/// step of keys at a time, the gradient against their scores.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn by_queries(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    out_gradient: &TileArg<'_>,
    logsumexp: &TileArg<'_>,
    delta: &TileArg<'_>,
    score_gradient: &TileArg<'_>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] rows: usize,
    #[comptime] block: usize,
) {
    let (query, key, value) = (query.tile(&space), key.tile(&space), value.tile(&space));
    let (out, out_gradient) = (out.tile(&space), out_gradient.tile(&space));
    let (logsumexp, delta) = (logsumexp.tile(&space), delta.tile(&space));
    let score_gradient = score_gradient.tile(&space);
    let bias = Procedural::new::<f32, Causal>(
        comptime!(space.space().subspace(&[Q, S])),
        Causal { keys, queries },
    )
    .tile_in::<f32>(&space);
    let instruction = comptime!(Instruction::Registers {
        config: RegisterBlock::new(rows * block)
    });
    for cube in space {
        for unit in cube {
            let mut delta_u = delta.at(&unit);
            delta_u.dot::<f32>(&out_gradient.at(&unit), &out.at(&unit));
            let (query_u, key_u, value_u) = (query.at(&unit), key.at(&unit), value.at(&unit));
            let (out_gradient_u, logsumexp_u) = (out_gradient.at(&unit), logsumexp.at(&unit));
            let (score_gradient_u, bias_u) = (score_gradient.at(&unit), bias.at(&unit));
            for step in unit.walk() {
                let bias_s = bias_u.at(&step);
                let (query_s, key_s) = (query_u.at(&step), key_u.at(&step));
                let (value_s, out_gradient_s) = (value_u.at(&step), out_gradient_u.at(&step));
                let mut score =
                    bias_s.accumulator::<f32>(&query_s, &key_s, instruction, Semiring::SUM_PROD);
                score.mma(&query_s, &key_s);
                score
                    .along(S)
                    .probabilities::<f32>(scale, &bias_s, &logsumexp_u);
                let mut gradient = bias_s.accumulator::<f32>(
                    &out_gradient_s,
                    &value_s,
                    instruction,
                    Semiring::SUM_PROD,
                );
                gradient.mma(&out_gradient_s, &value_s);
                gradient
                    .along(S)
                    .softmax_gradient::<f32>(&score.along(S), &delta_u);
                gradient.drained_into(&score_gradient_u.at(&step));
            }
        }
    }
}

/// Every unit owning `rows` keys, its scores held keys first: a step of queries at a time, the
/// gradient against their scores, each query's logsumexp and dot product read along the slices.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn by_keys(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out_gradient: &TileArg<'_>,
    logsumexp: &TileArg<'_>,
    delta: &TileArg<'_>,
    score_gradient: &TileArg<'_>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] rows: usize,
    #[comptime] block: usize,
) {
    let (query, key, value) = (query.tile(&space), key.tile(&space), value.tile(&space));
    let out_gradient = out_gradient.tile(&space);
    let (logsumexp, delta) = (logsumexp.tile(&space), delta.tile(&space));
    let score_gradient = score_gradient.tile(&space);
    let bias = Procedural::new::<f32, Causal>(
        comptime!(space.space().subspace(&[S, Q])),
        Causal { keys, queries },
    )
    .tile_in::<f32>(&space);
    let instruction = comptime!(Instruction::Registers {
        config: RegisterBlock::new(rows * block)
    });
    for cube in space {
        for unit in cube {
            let (query_u, key_u, value_u) = (query.at(&unit), key.at(&unit), value.at(&unit));
            let (out_gradient_u, logsumexp_u) = (out_gradient.at(&unit), logsumexp.at(&unit));
            let (delta_u, score_gradient_u, bias_u) =
                (delta.at(&unit), score_gradient.at(&unit), bias.at(&unit));
            for step in unit.walk() {
                let bias_s = bias_u.at(&step);
                let (query_s, key_s) = (query_u.at(&step), key_u.at(&step));
                let (value_s, out_gradient_s) = (value_u.at(&step), out_gradient_u.at(&step));
                let mut score =
                    bias_s.accumulator::<f32>(&key_s, &query_s, instruction, Semiring::SUM_PROD);
                score.mma(&key_s, &query_s);
                score
                    .along(Q)
                    .probabilities::<f32>(scale, &bias_s, &logsumexp_u.at(&step));
                let mut gradient = bias_s.accumulator::<f32>(
                    &value_s,
                    &out_gradient_s,
                    instruction,
                    Semiring::SUM_PROD,
                );
                gradient.mma(&value_s, &out_gradient_s);
                gradient
                    .along(Q)
                    .softmax_gradient::<f32>(&score.along(Q), &delta_u.at(&step));
                gradient.drained_into(&score_gradient_u.at(&step));
            }
        }
    }
}

/// Every unit owning `rows` queries, taking its scores into an online softmax a step of keys at
/// a time and leaving each query's logsumexp behind.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn logsumexp_of(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    logsumexp: &TileArg<'_>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] rows: usize,
    #[comptime] block: usize,
) {
    let (query, key, logsumexp) = (query.tile(&space), key.tile(&space), logsumexp.tile(&space));
    let bias = Procedural::new::<f32, Causal>(
        comptime!(space.space().subspace(&[Q, S])),
        Causal { keys, queries },
    )
    .tile_in::<f32>(&space);
    let instruction = comptime!(Instruction::Registers {
        config: RegisterBlock::new(rows * block)
    });
    for cube in space {
        for unit in cube {
            let (query_u, key_u, bias_u) = (query.at(&unit), key.at(&unit), bias.at(&unit));
            let mut softmax = OnlineSoftmax::<f32>::along(&bias_u, S);
            for step in unit.walk() {
                let bias_s = bias_u.at(&step);
                let (query_s, key_s) = (query_u.at(&step), key_u.at(&step));
                let mut score =
                    bias_s.accumulator::<f32>(&query_s, &key_s, instruction, Semiring::SUM_PROD);
                score.mma(&query_s, &key_s);
                softmax.step(&score, &bias_s, scale);
            }
            softmax.store_logsumexp(&bias_u, &mut logsumexp.at(&unit));
        }
    }
}

/// The softmax's quantities on the host, in `f64`: what each query's backward pass reads and
/// computes.
struct Reference {
    /// `[q]`: each query's logsumexp of its scaled scores.
    logsumexp: Vec<f64>,
    /// `[q, v]`: the forward's output.
    out: Vec<f64>,
    /// `[q]`: `Σ_v out_gradient · out`.
    delta: Vec<f64>,
    /// `[q, s]`: the gradient against the scores.
    score_gradient: Vec<f64>,
}

/// The shape of one case.
#[derive(Clone, Copy)]
struct Case {
    queries: usize,
    keys: usize,
    head: usize,
    value: usize,
    /// Queries or keys one unit owns.
    rows: usize,
    /// Keys or queries one step covers.
    block: usize,
    units: usize,
}

const CASE: Case = Case {
    queries: 16,
    keys: 24,
    head: 8,
    value: 8,
    rows: 2,
    block: 4,
    units: 4,
};

impl Case {
    fn scale(&self) -> f32 {
        1. / (self.head as f32).sqrt()
    }

    fn sees(&self, query: usize, key: usize) -> bool {
        key + self.queries <= query + self.keys
    }

    /// The inputs on `client`, with their host copies and everything the host computes from
    /// them.
    #[allow(clippy::type_complexity)]
    fn reference(&self, client: &Client) -> (Reference, [cubecl::std::tensor::TensorHandle; 4]) {
        let wobble =
            |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
        let input = |shape: [usize; 2], salt: usize| {
            TestInput::builder(client.clone(), Shape::new(shape))
                .dtype(f32::elem_type_native())
                .custom((0..shape[0] * shape[1]).map(|i| wobble(i, salt)).collect())
                .generate_with_f32_host_data()
        };
        let (query_handle, query) = input([self.queries, self.head], 1);
        let (key_handle, key) = input([self.keys, self.head], 2);
        let (value_handle, value) = input([self.keys, self.value], 3);
        let (out_gradient_handle, out_gradient) = input([self.queries, self.value], 4);
        let (queries, keys) = (self.queries, self.keys);
        let scale = self.scale() as f64;
        let score = |i: usize, j: usize| {
            (0..self.head)
                .map(|d| query.get_f32(&[i, d]) as f64 * key.get_f32(&[j, d]) as f64)
                .sum::<f64>()
                * scale
        };
        let logsumexp: Vec<f64> = (0..queries)
            .map(|i| {
                let seen: Vec<f64> = (0..keys)
                    .filter(|&j| self.sees(i, j))
                    .map(|j| score(i, j))
                    .collect();
                let max = seen
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .expect("every query sees a key");
                max + seen.iter().map(|s| (s - max).exp()).sum::<f64>().ln()
            })
            .collect();
        let probability = |i: usize, j: usize| match self.sees(i, j) {
            true => (score(i, j) - logsumexp[i]).exp(),
            false => 0.,
        };
        let out: Vec<f64> = (0..queries * self.value)
            .map(|at| {
                let (i, c) = (at / self.value, at % self.value);
                (0..keys)
                    .map(|j| probability(i, j) * value.get_f32(&[j, c]) as f64)
                    .sum()
            })
            .collect();
        let delta: Vec<f64> = (0..queries)
            .map(|i| {
                (0..self.value)
                    .map(|c| out_gradient.get_f32(&[i, c]) as f64 * out[i * self.value + c])
                    .sum()
            })
            .collect();
        let score_gradient = (0..queries * keys)
            .map(|at| {
                let (i, j) = (at / keys, at % keys);
                let probability_gradient: f64 = (0..self.value)
                    .map(|c| out_gradient.get_f32(&[i, c]) as f64 * value.get_f32(&[j, c]) as f64)
                    .sum();
                probability(i, j) * (probability_gradient - delta[i])
            })
            .collect();
        (
            Reference {
                logsumexp,
                out,
                delta,
                score_gradient,
            },
            [query_handle, key_handle, value_handle, out_gradient_handle],
        )
    }
}

/// A device tensor of `shape` holding `values`.
fn device(client: &Client, shape: Shape, values: &[f64]) -> cubecl::std::tensor::TensorHandle {
    TestInput::builder(client.clone(), shape)
        .dtype(f32::elem_type_native())
        .custom(values.iter().map(|&v| v as f32).collect())
        .generate_without_host_data()
}

/// A device tensor of `shape` holding zeros.
fn zeros(client: &Client, shape: Shape) -> cubecl::std::tensor::TensorHandle {
    TestInput::builder(client.clone(), shape)
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data()
}

/// `handle` bound as a tile over `axes`.
fn arg(handle: &cubecl::std::tensor::TensorHandle, axes: &[Axis]) -> TileArgLaunch {
    TileArgLaunch::new(
        handle.clone().binding().into_tensor_arg(),
        f32::elem_type_native(),
        1,
        TileSpec::direct(axes),
    )
}

/// Every value `read` gives at `[i, j]` of `shape` within `1e-4` of `expected` relative to one
/// plus its size.
fn assert_close(
    name: &str,
    read: impl Fn(usize, usize) -> f32,
    shape: [usize; 2],
    expected: &[f64],
) {
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let (got, want) = (read(i, j) as f64, expected[i * shape[1] + j]);
            assert!(
                (got - want).abs() <= 1e-4 * (1. + want.abs()),
                "{name}[{i}, {j}] = {got}, expected {want}"
            );
        }
    }
}

/// Units owning queries compute each query's dot product and the gradient against its scores as
/// the host does: the probabilities read back from the logsumexp are the forward's.
#[test]
fn units_owning_queries_take_the_softmax_gradient() {
    let client: Client = cubecl::test_device().client();
    let case = CASE;
    let (reference, [query, key, value, out_gradient]) = case.reference(&client);
    let (queries, keys) = (case.queries, case.keys);
    let out = device(&client, Shape::new([queries, case.value]), &reference.out);
    let logsumexp = device(&client, Shape::new([queries]), &reference.logsumexp);
    let delta = zeros(&client, Shape::new([queries]));
    let score_gradient = zeros(&client, Shape::new([queries, keys]));
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, case.head), (V, case.value)]),
            Levels::leaf(&[
                (Q, case.rows),
                (S, case.block),
                (D, case.head),
                (V, case.value),
            ])
            .walk_every(&[S])
            .units_distributed(Q, case.units)
            .cubes(&[Q])
            .build(),
        ),
        Form::Static,
    );
    by_queries::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&query, &[Q, D]),
        arg(&key, &[S, D]),
        arg(&value, &[S, V]),
        arg(&out, &[Q, V]),
        arg(&out_gradient, &[Q, V]),
        arg(&logsumexp, &[Q]),
        arg(&delta, &[Q]),
        arg(&score_gradient, &[Q, S]),
        keys as u32,
        queries as u32,
        case.scale(),
        launcher.partitioning_arg(),
        case.rows,
        case.block,
    );
    let delta = HostData::from_tensor_handle(&client, delta, HostDataType::F32);
    let score_gradient = HostData::from_tensor_handle(&client, score_gradient, HostDataType::F32);
    assert_close(
        "delta",
        |i, _| delta.get_f32(&[i]),
        [queries, 1],
        &reference.delta,
    );
    assert_close(
        "score_gradient",
        |i, j| score_gradient.get_f32(&[i, j]),
        [queries, keys],
        &reference.score_gradient,
    );
}

/// Units owning keys, their scores held keys first, compute the same gradient: each query's
/// logsumexp and dot product broadcast along the slices rather than across them.
#[test]
fn units_owning_keys_take_the_softmax_gradient_keys_first() {
    let client: Client = cubecl::test_device().client();
    let case = CASE;
    let (reference, [query, key, value, out_gradient]) = case.reference(&client);
    let (queries, keys) = (case.queries, case.keys);
    let logsumexp = device(&client, Shape::new([queries]), &reference.logsumexp);
    let delta = device(&client, Shape::new([queries]), &reference.delta);
    let score_gradient = zeros(&client, Shape::new([keys, queries]));
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(S, keys), (Q, queries), (D, case.head), (V, case.value)]),
            Levels::leaf(&[
                (S, case.rows),
                (Q, case.block),
                (D, case.head),
                (V, case.value),
            ])
            .walk_every(&[Q])
            .units_distributed(S, case.units)
            .cubes(&[S])
            .build(),
        ),
        Form::Static,
    );
    by_keys::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&query, &[Q, D]),
        arg(&key, &[S, D]),
        arg(&value, &[S, V]),
        arg(&out_gradient, &[Q, V]),
        arg(&logsumexp, &[Q]),
        arg(&delta, &[Q]),
        arg(&score_gradient, &[S, Q]),
        keys as u32,
        queries as u32,
        case.scale(),
        launcher.partitioning_arg(),
        case.rows,
        case.block,
    );
    let score_gradient = HostData::from_tensor_handle(&client, score_gradient, HostDataType::F32);
    let transposed: Vec<f64> = (0..keys * queries)
        .map(|at| reference.score_gradient[(at % queries) * keys + at / queries])
        .collect();
    assert_close(
        "score_gradient",
        |i, j| score_gradient.get_f32(&[i, j]),
        [keys, queries],
        &transposed,
    );
}

/// The logsumexp an online softmax leaves behind is the host's: what a backward pass reads its
/// probabilities back from.
#[test]
fn an_online_softmax_leaves_its_logsumexp() {
    let client: Client = cubecl::test_device().client();
    let case = CASE;
    let (reference, [query, key, _, _]) = case.reference(&client);
    let (queries, keys) = (case.queries, case.keys);
    let logsumexp = zeros(&client, Shape::new([queries]));
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, case.head)]),
            Levels::leaf(&[(Q, case.rows), (S, case.block), (D, case.head)])
                .walk_every(&[S])
                .units_distributed(Q, case.units)
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    logsumexp_of::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        arg(&query, &[Q, D]),
        arg(&key, &[S, D]),
        arg(&logsumexp, &[Q]),
        keys as u32,
        queries as u32,
        case.scale(),
        launcher.partitioning_arg(),
        case.rows,
        case.block,
    );
    let logsumexp = HostData::from_tensor_handle(&client, logsumexp, HostDataType::F32);
    assert_close(
        "logsumexp",
        |i, _| logsumexp.get_f32(&[i]),
        [queries, 1],
        &reference.logsumexp,
    );
}
