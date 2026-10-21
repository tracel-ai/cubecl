//! Operands that carry their indices ([`TileArgLaunch::indexed`]): a table of `u32` entries rides
//! the operand like a scale, and the tile reads it where its window is cut.
//!
//! The kernels here read no table and route no walk. Routed rows and grouped tiles are one kernel
//! under two tables: the weights' expert is their own, read at the step that cuts the rows, one
//! entry a row or one a tile of rows. Ragged runs are a plain sum over the run the activations
//! place themselves at.
#![allow(non_snake_case)]

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    bytes::Bytes, client::Client, prelude::*, quant::scheme::QuantValue, std::tensor::TensorHandle,
    zspace::Shape,
};
use cubecl_tile::kind::Boundary;
use cubecl_tile::launch::Refusal;
use cubecl_tile::*;

/// Row: a token, or a token's slot.
const M: Axis = Axis(0);
/// Output feature.
const N: Axis = Axis(1);
/// Contracted feature.
const K: Axis = Axis(2);
/// Which expert. Three of them, and the space says so; no level cuts it.
const EXPERT: Axis = Axis(3);

const ROWS: usize = 4;
const FEATURES: usize = 4;
const EXPERTS: usize = 3;

/// Which expert each row routes to. Distinct enough that reusing one row's expert for another
/// cannot land on the right answer.
const ROUTES: [u32; ROWS] = [2, 0, 2, 1];

const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `x[m, k]`.
fn activations() -> Vec<f32> {
    (0..ROWS * FEATURES).map(|i| (i % 5) as f32).collect()
}

/// `w[e, k, n]`, expert `e` scaling by `e + 1`, so contracting a row against the wrong slab is off
/// by a whole factor.
fn weights() -> Vec<f32> {
    (0..EXPERTS * FEATURES * FEATURES)
        .map(|i| {
            let e = i / (FEATURES * FEATURES);
            ((e + 1) * (i % 3 + 1)) as f32
        })
        .collect()
}

/// Each row against the slab `expert_of(m)` names, summed on the host; zeros where it names none.
fn expected(expert_of: impl Fn(usize) -> usize, w: &[f32]) -> Vec<Vec<f32>> {
    let x = activations();
    (0..ROWS)
        .map(|m| {
            let e = expert_of(m);
            (0..FEATURES)
                .map(|n| match e < EXPERTS {
                    true => (0..FEATURES)
                        .map(|k| x[m * FEATURES + k] * w[(e * FEATURES + k) * FEATURES + n])
                        .sum(),
                    false => 0.0,
                })
                .collect()
        })
        .collect()
}

fn assert_cells(got: &HostData, want: &[Vec<f32>], what: &str) {
    for (m, row) in want.iter().enumerate() {
        for (n, cell) in row.iter().enumerate() {
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - cell).abs() < 1e-3,
                "{what}: row {m} feature {n}: got {have}, want {cell}"
            );
        }
    }
}

/// A `u32` table of `entries` over `keys`, one dim each, shaped `shape`.
fn table(client: &Client, entries: &[u32], shape: Shape, keys: &[Axis]) -> TileArgLaunch {
    let handle = TestInput::builder(client.clone(), shape)
        .dtype(u32::elem_type_native())
        .custom(entries.iter().map(|&e| e as f32).collect())
        .generate_without_host_data();
    TileArgLaunch::new(
        handle.binding().into_tensor_arg(),
        u32::elem_type_native(),
        1,
        TileSpec::direct(keys),
    )
}

/// A `[ROWS, FEATURES]` output filled with a value no product lands on, so a launch that never
/// ran fails rather than passes.
fn unwritten_output(client: &Client) -> TensorHandle {
    TestInput::builder(client.clone(), Shape::new([ROWS, FEATURES]))
        .dtype(f32::elem_type_native())
        .custom(vec![-1.0; ROWS * FEATURES])
        .generate_without_host_data()
}

fn f32_arg(handle: TensorHandle, spec: TileSpec) -> TileArgLaunch {
    TileArgLaunch::new(
        handle.binding().into_tensor_arg(),
        f32::elem_type_native(),
        1,
        spec,
    )
}

/// `out[rows] = x[rows] · W[e]`, `e` the weights' own: the dense register-block product, word
/// for word. Whether a step is a row or a tile of rows is the partitioning's; which slab it reads
/// is the table the weights carry.
#[cube(launch)]
fn indexed_matmul_kernel(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] rows: Level,
) {
    let x = x.tile(&space);
    let w = w.tile(&space);
    let out = out.tile(&space);

    for tile in space.over(&rows) {
        let mut o = out
            .at(&tile)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        o.mm(&x.at(&tile), &w.at(&tile));
    }
}

/// [`indexed_matmul_kernel`] with the weights staged in shared memory over the walk of rows: the
/// weights span no axis that walk steps, so only their table tells the plan to refill the stage.
#[cube(launch)]
fn indexed_staged_matmul_kernel(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] rows: Level,
) {
    let x = x.tile(&space);
    let w = w.tile(&space);
    let out = out.tile(&space);

    let tiles = space.over(&rows);
    let mut stages = Stages::smem_single(&tiles, &w, StageStorage::Strided, 1usize, None);
    stages.pipelined(tiles, |slot, tile| {
        let mut o = out
            .at(tile)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        slot.consume(|w_s| {
            o.mm(&x.at(tile), w_s);
        });
    });
}

/// [`indexed_matmul_kernel`] that skips a step whose entry names no slab: what a grouped product
/// does with a padding tile, which it would otherwise contract against a clamped slab.
#[cube(launch)]
fn indexed_skipping_matmul_kernel(
    x: &TileArg<'_>,
    w: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] rows: Level,
) {
    let x = x.tile(&space);
    let w = w.tile(&space);
    let out = out.tile(&space);

    for tile in space.over(&rows) {
        let w_t = w.at(&tile);
        if w_t.entry_in_range() {
            let mut o = out
                .at(&tile)
                .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
            o.mm(&x.at(&tile), &w_t);
        }
    }
}

/// How the weights are read: where they lie, or staged per step of rows.
#[derive(Clone, Copy)]
enum Reading {
    InPlace,
    Staged,
    /// In place, skipping a step whose entry names no slab.
    Skipping,
}

/// The rows in tiles of `tile`, the weights `[EXPERT, K, N]` keyed by the row tile: `entries`
/// holds one expert per tile.
fn run(entries: &[u32], tile: usize, weights_spec: TileSpec, reading: Reading) -> HostData {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, ROWS), (N, FEATURES), (K, FEATURES), (EXPERT, EXPERTS)]),
            Levels::leaf(&[(M, tile)]).walk_every(&[M]).build(),
        ),
        Form::Static,
    );
    let f32_ty = f32::elem_type_native();
    let (x, _) = TestInput::builder(client.clone(), Shape::new([ROWS, FEATURES]))
        .dtype(f32_ty)
        .custom(activations())
        .generate_with_f32_host_data();
    let (w, _) = TestInput::builder(client.clone(), Shape::new([EXPERTS, FEATURES, FEATURES]))
        .dtype(f32_ty)
        .custom(weights())
        .generate_with_f32_host_data();
    let routes = table(&client, entries, Shape::new([entries.len()]), &[M]);
    let out = unwritten_output(&client);

    let x = f32_arg(x, TileSpec::direct(&[M, K]));
    let w = f32_arg(w, weights_spec)
        .indexed(routes, Indexing::Coordinate(EXPERT))
        .unwrap();
    let o = f32_arg(out.clone(), TileSpec::direct(&[M, N]));
    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    let level = launcher.partitioning().level(0);
    match reading {
        Reading::InPlace => indexed_matmul_kernel::launch(
            &client,
            count,
            dim,
            x,
            w,
            o,
            launcher.partitioning_arg(),
            level,
        ),
        Reading::Staged => indexed_staged_matmul_kernel::launch(
            &client,
            count,
            dim,
            x,
            w,
            o,
            launcher.partitioning_arg(),
            level,
        ),
        Reading::Skipping => indexed_skipping_matmul_kernel::launch(
            &client,
            count,
            dim,
            x,
            w,
            o,
            launcher.partitioning_arg(),
            level,
        ),
    }

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

fn weights_spec() -> TileSpec {
    TileSpec::direct(&[EXPERT, K, N])
}

/// One entry a row: each row's expert. Two tables over the same weights, so the answer follows
/// the table; an index that did nothing would read the first slab under both.
#[test]
fn a_routed_operand_contracts_each_row_against_the_expert_its_table_names() {
    for routes in [ROUTES, [1, 1, 0, 2]] {
        let got = run(&routes, 1, weights_spec(), Reading::InPlace);
        let want = expected(|m| routes[m] as usize, &weights());
        assert_cells(&got, &want, "routed");
    }
}

/// An entry past the stack reads the last slab, as a routed walk clamps: the table is device
/// data, and a read past the weights would land outside their buffer.
#[test]
fn a_route_past_the_last_expert_clamps_to_it() {
    let got = run(&[99, 0, 2, 1], 1, weights_spec(), Reading::InPlace);
    let want = expected(|m| [EXPERTS - 1, 0, 2, 1][m], &weights());
    assert_cells(&got, &want, "clamped");
}

/// Weights that check their expert axis read zero past it instead: the row a padded route names
/// drains the identity, with no branch in the kernel.
#[test]
fn a_route_past_the_last_expert_reads_zero_where_the_operand_checks_it() {
    let checked = weights_spec().boundaries(&[Some(Boundary::Zero), None, None]);
    let got = run(&[99, 0, 2, 1], 1, checked, Reading::InPlace);
    let want = expected(|m| [99, 0, 2, 1][m], &weights());
    assert_cells(&got, &want, "checked");
}

/// A kernel asks the tile whether its entry named a slab, and skips the row that names none: the
/// row keeps the value the output was filled with, every other row its product.
#[test]
fn a_kernel_skips_the_row_whose_route_names_no_expert() {
    let routes = [99, 0, 2, 1];
    let got = run(&routes, 1, weights_spec(), Reading::Skipping);
    let mut want = expected(|m| routes[m] as usize, &weights());
    want[0] = vec![-1.0; FEATURES];
    assert_cells(&got, &want, "skipped");
}

/// One entry a tile of two rows: each tile's group, the same kernel as routed rows. Tile 0 reads
/// expert 2 and tile 1 expert 0, so a per-row reading of this table would give rows 1 and 3 the
/// wrong slab.
#[test]
fn a_grouped_operand_contracts_each_tile_of_rows_against_its_group() {
    let groups = [2u32, 0];
    let got = run(&groups, 2, weights_spec(), Reading::InPlace);
    let want = expected(|m| groups[m / 2] as usize, &weights());
    assert_cells(&got, &want, "grouped");
}

/// The grouped weights staged per tile of rows: the walk steps rows, which the weights do not
/// span, so a plan blind to their table would fill the first group's slab once and reuse it.
#[test]
fn a_grouped_operand_stages_the_slab_of_each_tile() {
    let groups = [1u32, 2];
    let got = run(&groups, 2, weights_spec(), Reading::Staged);
    let want = expected(|m| groups[m / 2] as usize, &weights());
    assert_cells(&got, &want, "grouped, staged");
}

/// `out[rows] = x[rows] · (W[e] ⊗ s[e])` with `W` packed eight bits a field, four to a word, and
/// one scale per expert and contracted feature: the routed weights are quantized, and the scales
/// they carry move to the same expert.
#[test]
fn a_routed_packed_operand_reads_its_scales_at_the_same_expert() {
    let field = QuantValue::Q8S;
    let bits = field.size_bits();
    let factor = 32 / bits;
    let client = cubecl::test_device().client();
    let max = client.properties().hardware.max_vector_size;
    if factor > max {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max}, below the {factor}-value word"
        ))
        .enforce();
        return;
    }

    // Small signed values, distinct per expert so a slab read off the wrong expert shows.
    let values: Vec<i32> = (0..EXPERTS * FEATURES * FEATURES)
        .map(|i| {
            let e = i / (FEATURES * FEATURES);
            (i % 7) as i32 - 3 + e as i32
        })
        .collect();
    let mask = (1u32 << bits) - 1;
    let words: Vec<u32> = values
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &v)| acc | ((v as u32 & mask) << (j * bits)))
        })
        .collect();
    let w = TensorHandle::new_contiguous(
        vec![EXPERTS, FEATURES, FEATURES],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let scales: Vec<f32> = (0..EXPERTS * FEATURES)
        .map(|i| 0.5 * (i / FEATURES + 1) as f32 + 0.25 * (i % FEATURES) as f32)
        .collect();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, ROWS), (N, FEATURES), (K, FEATURES), (EXPERT, EXPERTS)]),
            Levels::leaf(&[(M, 1)]).walk_every(&[M]).build(),
        ),
        Form::Static,
    );
    let f32_ty = f32::elem_type_native();
    let (x, _) = TestInput::builder(client.clone(), Shape::new([ROWS, FEATURES]))
        .dtype(f32_ty)
        .custom(activations())
        .generate_with_f32_host_data();
    let (s, _) = TestInput::builder(client.clone(), Shape::new([EXPERTS, FEATURES]))
        .dtype(f32_ty)
        .custom(scales.clone())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), Shape::new([ROWS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; ROWS * FEATURES])
        .generate_without_host_data();

    let w = TileArgLaunch::new(
        w.binding().into_tensor_arg(),
        u32::elem_type_native(),
        1,
        TileSpec::direct(&[EXPERT, K, N]).packed(field),
    )
    .scales(f32_arg(s, TileSpec::direct(&[EXPERT, K])))
    .indexed(
        table(&client, &ROUTES, Shape::new([ROWS]), &[M]),
        Indexing::Coordinate(EXPERT),
    )
    .unwrap();

    indexed_matmul_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        f32_arg(x, TileSpec::direct(&[M, K])),
        w,
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            f32_ty,
            factor,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    let dequantized: Vec<f32> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let (e, k) = (i / (FEATURES * FEATURES), i / FEATURES % FEATURES);
            v as f32 * scales[e * FEATURES + k]
        })
        .collect();
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = expected(|m| ROUTES[m] as usize, &dequantized);
    assert_cells(&got, &want, "routed, packed");
}

/// Which group.
const G: Axis = Axis(0);
/// The packed row: every group's rows, end to end in one buffer.
const P: Axis = Axis(1);
/// The feature a row carries.
const D: Axis = Axis(2);

const GROUPS: usize = 4;
/// Rows of every group, end to end, and rows past the last group's end.
const PACKED: usize = 16;
/// Rows in blocks of four: groups start and end off a block's edge.
const BLOCK: usize = 4;

/// `out[g] = Σ packed[p]` over group `g`'s rows: the plain blocked sum over the rows the
/// activations place themselves at. The walk is the run's, so a group of no rows takes no step.
#[cube(launch)]
fn indexed_ragged_sum_kernel<E: Numeric>(
    packed: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] group: Level,
    #[comptime] block: Level,
    #[define(E)] _dtype: ElemType,
) {
    let packed = packed.tile(&space);
    let out = out.tile(&space);

    for g in space.over(&group) {
        let rows = packed.at(&g);
        let mut total = out.at(&g);
        total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));
        for blk in rows.over(&block) {
            let mut cell = total.at(&blk);
            cell.reduce_axis_accumulate(&rows.at(&blk), comptime!(Monoid::Sum));
        }
    }
}

/// [`indexed_ragged_sum_kernel`] with each block of a group's rows staged in shared memory before
/// it is summed, `depth` stages in flight. The blocks are the group's walk over its run.
#[cube(launch)]
fn indexed_staged_ragged_sum_kernel<E: Numeric>(
    packed: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] group: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let packed = packed.tile(&space);
    let out = out.tile(&space);

    for g in space.over(&group) {
        let rows = packed.at(&g);
        let mut total = out.at(&g);
        total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));
        let blocks = g.walk_within(&rows);
        let mut stages = Stages::smem_single(&blocks, &rows, StageStorage::Strided, depth, None);
        stages.pipelined(blocks, |slot, blk| {
            let mut cell = total.at(blk);
            slot.consume(|rows_s| {
                cell.reduce_axis_accumulate(rows_s, comptime!(Monoid::Sum));
            });
        });
    }
}

/// Every packed cell distinct, so summing a neighbouring group's row cannot land on the right
/// answer by accident.
fn packed_values() -> Vec<f32> {
    (0..PACKED * FEATURES)
        .map(|i| ((i / FEATURES) * 10 + i % FEATURES) as f32)
        .collect()
}

/// How the packed rows lie: each row's features together, or each feature's rows together, the
/// rows along the innermost dim, as a transposed operand reads them.
#[derive(Clone, Copy, Debug)]
enum Rows {
    Lined,
    Transposed,
}

/// How the ragged sum reads each block: where it lies, or staged with this many stages in flight.
#[derive(Clone, Copy, Debug)]
enum BlockRead {
    InPlace,
    Staged(usize),
}

/// The sum over the groups `ends` cuts the rows into, `ends` each group's cumulative end row.
fn run_ragged(ends: &[u32], rows: Rows, read: BlockRead) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(G, GROUPS), (P, PACKED), (D, FEATURES)]),
            Levels::leaf(&[(G, 1), (P, BLOCK)])
                .walk_every(&[P])
                .walk_every(&[G])
                .build(),
        ),
        // The rows a group walks are its run's: an extent read at runtime.
        Form::DynamicAlong(&[P]),
    );
    let values = packed_values();
    let (shape, values, spec) = match rows {
        Rows::Lined => ([PACKED, FEATURES], values, TileSpec::direct(&[P, D])),
        Rows::Transposed => {
            let transposed = (0..FEATURES * PACKED)
                .map(|i| values[(i % PACKED) * FEATURES + i / PACKED])
                .collect();
            ([FEATURES, PACKED], transposed, TileSpec::direct(&[D, P]))
        }
    };
    let (packed, _) = TestInput::builder(client.clone(), Shape::new(shape))
        .dtype(f32_ty)
        .custom(values)
        .generate_with_f32_host_data();
    // Poisoned, not zeroed: the kernel owes every cell a value, the empty group's included.
    let out = TestInput::builder(client.clone(), Shape::new([GROUPS, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; GROUPS * FEATURES])
        .generate_without_host_data();

    let packed = f32_arg(packed, spec)
        .indexed(
            table(&client, ends, Shape::new([ends.len()]), &[G]),
            Indexing::Run(P),
        )
        .unwrap();
    let total = f32_arg(out.clone(), TileSpec::direct(&[G, D]));
    let (count, dim) = (launcher.cube_count(), launcher.cube_dim());
    let (group, block) = (
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
    );
    match read {
        BlockRead::InPlace => indexed_ragged_sum_kernel::launch(
            &client,
            count,
            dim,
            packed,
            total,
            launcher.partitioning_arg(),
            group,
            block,
            f32_ty,
        ),
        BlockRead::Staged(depth) => indexed_staged_ragged_sum_kernel::launch(
            &client,
            count,
            dim,
            packed,
            total,
            launcher.partitioning_arg(),
            group,
            depth,
            f32_ty,
        ),
    }

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// Group `g`'s rows, from the end before it (`0` for the first) to its own, summed on the host;
/// an end before its start sums nothing.
fn expected_ragged(ends: &[u32]) -> Vec<Vec<f32>> {
    let values = packed_values();
    (0..GROUPS)
        .map(|g| {
            let start = if g == 0 { 0 } else { ends[g - 1] as usize };
            (0..FEATURES)
                .map(|d| {
                    (start..ends[g] as usize)
                        .map(|p| values[p * FEATURES + d])
                        .sum()
                })
                .collect()
        })
        .collect()
}

fn assert_ragged(got: &HostData, ends: &[u32], what: &str) {
    for (g, row) in expected_ragged(ends).iter().enumerate() {
        for (d, cell) in row.iter().enumerate() {
            assert_eq!(
                got.get_f32(&[g, d]),
                *cell,
                "{what}: group {g} feature {d} under ends {ends:?}"
            );
        }
    }
}

/// Each group's run off its cumulative end: runs of 2, 0, 5 and 3 rows, then 3, 0, 3 and 4. The
/// answer follows the table, which it could not if the run did nothing, since every group would
/// sum from row zero under both.
#[test]
fn a_ragged_operand_sums_each_group_over_its_own_rows() {
    for ends in [[2u32, 2, 7, 10], [3, 3, 6, 10]] {
        assert_ragged(
            &run_ragged(&ends, Rows::Lined, BlockRead::InPlace),
            &ends,
            "lined",
        );
    }
}

/// The rows along the innermost dim, read one value a line: a run placed there is an element
/// offset like any other, as a transposed operand's rows are.
#[test]
fn a_ragged_operand_runs_along_its_innermost_dim() {
    let ends = [2u32, 2, 7, 10];
    assert_ragged(
        &run_ragged(&ends, Rows::Transposed, BlockRead::InPlace),
        &ends,
        "transposed",
    );
}

/// The ragged sum with each block staged first, one or two stages in flight: the walk over a
/// run proves every block but its last inside the run, which is staged unchecked, and the last
/// reads zero past the run's end. Runs of 2, 0, 5 and 3 rows; 3, 0, 3 and 4, the last one whole
/// block; and 2, 0, 13 and 1, the third four blocks long, enough for two stages to have laps
/// with nothing to check.
#[test]
fn a_ragged_operand_stages_each_block_of_its_run() {
    for ends in [[2u32, 2, 7, 10], [3, 3, 6, 10], [2, 2, 15, 16]] {
        for rows in [Rows::Lined, Rows::Transposed] {
            for depth in [1, 2] {
                let got = run_ragged(&ends, rows, BlockRead::Staged(depth));
                assert_ragged(&got, &ends, &format!("staged {depth} deep, {rows:?}"));
            }
        }
    }
}

/// A table whose second group ends before it starts: that group is empty, not a run of some
/// four billion rows, and its neighbours sum as stated.
#[test]
fn a_run_that_ends_before_it_starts_is_empty() {
    let ends = [4u32, 2, 7, 10];
    let got = run_ragged(&ends, Rows::Lined, BlockRead::InPlace);
    for d in 0..FEATURES {
        assert_eq!(got.get_f32(&[1, d]), 0.0, "group 1 feature {d}");
    }
    let want = expected_ragged(&ends);
    for g in [0, 2, 3] {
        for (d, cell) in want[g].iter().enumerate() {
            assert_eq!(got.get_f32(&[g, d]), *cell, "group {g} feature {d}");
        }
    }
}

/// `out = Σ packed[p]` over the one group's rows: no level cuts the group, so the rows hold one
/// run, read as the tile is served, and the walk is that run's.
#[cube(launch)]
fn indexed_whole_sum_kernel<E: Numeric>(
    packed: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] block: Level,
    #[define(E)] _dtype: ElemType,
) {
    let rows = packed.tile(&space);
    let mut total = out.tile(&space);
    total.init::<E>(Monoid::identity::<E>(comptime!(Monoid::Sum)));
    for blk in rows.over(&block) {
        let mut cell = total.at(&blk);
        cell.reduce_axis_accumulate(&rows.at(&blk), comptime!(Monoid::Sum));
    }
}

/// A single group, which no level cuts: its run is the table's one entry, seven of the sixteen
/// rows, so the nine past it never reach the sum.
#[test]
fn a_table_no_level_cuts_places_its_one_run() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(G, 1), (P, PACKED), (D, FEATURES)]),
            Levels::leaf(&[(P, BLOCK)]).walk_every(&[P]).build(),
        ),
        Form::DynamicAlong(&[P]),
    );
    let (packed, _) = TestInput::builder(client.clone(), Shape::new([PACKED, FEATURES]))
        .dtype(f32_ty)
        .custom(packed_values())
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), Shape::new([1, FEATURES]))
        .dtype(f32_ty)
        .custom(vec![-1.0; FEATURES])
        .generate_without_host_data();
    let ends = [7u32];

    indexed_whole_sum_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        f32_arg(packed, TileSpec::direct(&[P, D]))
            .indexed(
                table(&client, &ends, Shape::new([1]), &[G]),
                Indexing::Run(P),
            )
            .unwrap(),
        f32_arg(out.clone(), TileSpec::direct(&[G, D])),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let values = packed_values();
    for d in 0..FEATURES {
        let want: f32 = (0..7).map(|p| values[p * FEATURES + d]).sum();
        assert_eq!(got.get_f32(&[0, d]), want, "feature {d}");
    }
}

/// Column tiles of the features: the walk below a group cuts them beside the rows.
const COLUMN: usize = 2;

/// Counts, for every group and column tile, the steps the walk over the group's run takes there.
/// The rows (`x`, `[P]`) span no column, so a walk over their own box would step column tile `0`
/// alone; the region's walk steps every column tile, each over the run the rows hold.
#[cube(launch)]
fn run_steps_kernel(
    x: &TileArg<'_>,
    steps: &mut Tensor<f32>,
    space: Partitioning,
    #[comptime] group: Level,
    #[comptime] columns: usize,
) {
    let x = x.tile(&space);
    for g in space.over(&group) {
        let x_g = x.at(&g);
        for blk in g.walk_within(&x_g) {
            let at = g.coord(G) * columns + blk.coord(D);
            steps[at] += 1.0;
        }
    }
}

/// Each group walks its own run's blocks in every column tile: 1, 0, 2 and 1 blocks of four rows
/// for runs of 2, 0, 5 and 3 rows.
#[test]
fn a_regions_walk_steps_the_run_its_operand_holds_over_every_axis() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let columns = FEATURES / COLUMN;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(G, GROUPS), (P, PACKED), (D, FEATURES)]),
            Levels::leaf(&[(G, 1), (P, BLOCK), (D, COLUMN)])
                .walk_every(&[P, D])
                .walk_every(&[G])
                .build(),
        ),
        Form::DynamicAlong(&[P]),
    );
    let (x, _) = TestInput::builder(client.clone(), Shape::new([PACKED]))
        .dtype(f32_ty)
        .custom(vec![1.0; PACKED])
        .generate_with_f32_host_data();
    let counted = TestInput::builder(client.clone(), Shape::new([GROUPS * columns]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let ends = [2u32, 2, 7, 10];

    run_steps_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        f32_arg(x, TileSpec::direct(&[P]))
            .indexed(
                table(&client, &ends, Shape::new([GROUPS]), &[G]),
                Indexing::Run(P),
            )
            .unwrap(),
        counted.clone().binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        columns,
    );

    let got = HostData::from_tensor_handle(&client, counted, HostDataType::F32);
    for (g, blocks) in [1.0, 0.0, 2.0, 1.0].into_iter().enumerate() {
        for c in 0..columns {
            assert_eq!(
                got.get_f32(&[g * columns + c]),
                blocks,
                "group {g} column tile {c}"
            );
        }
    }
}

/// The table's statement is checked where it is bound, on the caller's thread.
mod refused {
    use super::*;

    /// The weights, read four values a line along `N`.
    fn weights(client: &Client) -> TileArgLaunch {
        let handle = TestInput::builder(client.clone(), Shape::new([EXPERTS, FEATURES, FEATURES]))
            .dtype(f32::elem_type_native())
            .zeros()
            .generate_without_host_data();
        TileArgLaunch::new(
            handle.binding().into_tensor_arg(),
            f32::elem_type_native(),
            FEATURES,
            TileSpec::direct(&[EXPERT, K, N]),
        )
    }

    fn refusal(table: TileArgLaunch, indexing: Indexing) -> Refusal {
        let client = cubecl::test_device().client();
        match weights(&client).indexed(table, indexing) {
            Ok(_) => panic!("the statement was taken"),
            Err(refusal) => refusal,
        }
    }

    #[test]
    fn a_table_of_floats() {
        let client = cubecl::test_device().client();
        let handle = TestInput::builder(client.clone(), Shape::new([ROWS]))
            .dtype(f32::elem_type_native())
            .zeros()
            .generate_without_host_data();
        let floats = f32_arg(handle, TileSpec::direct(&[M]));
        assert_eq!(
            refusal(floats, Indexing::Coordinate(EXPERT)),
            Refusal::IndexTableNotU32
        );
    }

    #[test]
    fn a_table_of_more_dims_than_keys() {
        let client = cubecl::test_device().client();
        let two_dims = table(&client, &[0; ROWS], Shape::new([2, 2]), &[M]);
        assert_eq!(
            refusal(two_dims, Indexing::Coordinate(EXPERT)),
            Refusal::IndexTableRank { keys: 1, rank: 2 }
        );
    }

    /// The weights read four values a line along `N`: a coordinate there would split a line.
    #[test]
    fn an_entry_placing_the_innermost_axis_read_in_lines() {
        let client = cubecl::test_device().client();
        let routes = table(&client, &ROUTES, Shape::new([ROWS]), &[M]);
        assert_eq!(
            refusal(routes, Indexing::Coordinate(N)),
            Refusal::IndexAlongALine(N)
        );
    }

    #[test]
    fn an_entry_placing_an_axis_the_operand_does_not_span() {
        let client = cubecl::test_device().client();
        let routes = table(&client, &ROUTES, Shape::new([ROWS]), &[M]);
        assert_eq!(
            refusal(routes, Indexing::Coordinate(M)),
            Refusal::IndexOfAnAxisNotSpanned(M)
        );
    }

    #[test]
    fn a_run_keyed_by_two_axes() {
        let client = cubecl::test_device().client();
        let ends = table(&client, &[0; 4], Shape::new([2, 2]), &[M, N]);
        assert_eq!(
            refusal(ends, Indexing::Run(EXPERT)),
            Refusal::RunKeyedByMany(2)
        );
    }
}
