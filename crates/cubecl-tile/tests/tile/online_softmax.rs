//! An attention whose softmax runs through [`OnlineSoftmax`] where its holder keeps the score: each unit owning
//! its rows in registers, or each plane owning its rows in its own window, from loading to
//! writing.
//!
//! A unit reads its query, keys and values where they lie, holds its score in a register block,
//! and contracts the probabilities it just computed with the values without leaving registers.
//!
//! One partitioning spans both contractions: `score = q · kᵀ` contracts `D` and `out = p · v`
//! contracts `S`, and each walks its own axes of the fragment level, the other's routed to one
//! step. The plane stages its query once and its keys and values a block at a time, in stages of
//! its own met on `sync_plane`; the score lands in the plane's own window, the softmax takes it
//! in there, and the output's fragments are corrected through the plane's own scratch.
//!
//! The same kernel serves planes that split the keys of the same queries: the partitioning says
//! so, each plane walks its share, the softmax normalizer merges the planes' states
//! ([`OnlineSoftmax::normalizer`]), and the cube's output sums their products before writing
//! them ([`PlanesOutput`](cubecl_tile::kind::PlanesOutput)).

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Axis, Instruction, Levels, MmaOperands, Partitioning, RegisterBlock, Scratch, Semiring, Space,
    StageStorage, Stages, Tile, TileArg, TileArgLaunch, TileSpec,
    ops::{matmul::MmaIo, softmax::OnlineSoftmax},
    procedural::{Procedural, Reads, Recipe, RecipeCoords, RecipeExpand},
};

use super::relay::relays;
use super::{Form, implied};

const Q: Axis = Axis(0);
const S: Axis = Axis(1);
const D: Axis = Axis(2);
const V: Axis = Axis(3);

/// The bias a query's scores are summed with: zero at a key it sees, the minimum value past the
/// attended keys and, under `causal`, past its own position aligned at the bottom right.
#[derive(CubeType, Clone)]
struct Attended {
    keys: u32,
    queries: u32,
    #[cube(comptime)]
    causal: bool,
}

#[cube]
impl Recipe<f32> for Attended {
    fn evaluate(&self, coordinates: &RecipeCoords) -> f32 {
        let s = coordinates.along(S);
        let mut seen = s < self.keys;
        if self.causal {
            seen = seen && s + self.queries <= coordinates.along(Q) + self.keys;
        }
        select(seen, 0.0, f32::min_value())
    }
}

impl Reads for AttendedExpand {
    fn reads(&self, _scope: &cubecl::ir::Scope, axis: Axis) -> bool {
        axis == S || (self.causal && axis == Q)
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn plane_attention<E: Float>(
    q: &TileArg<'_>,
    k: &TileArg<'_>,
    v: &TileArg<'_>,
    out: &TileArg<'_>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] causal: bool,
    #[comptime] block_keys: usize,
    #[define(E)] _dtype: ElemType,
) {
    let q = q.tile(&space);
    // Past the attended keys a read is zero: a masked probability never meets a stale value.
    let k = k.tile(&space).within(S, 0, keys as usize);
    let v = v.tile(&space).within(S, 0, keys as usize);
    let out = out.tile(&space);
    let bias = Procedural::new::<f32, Attended>(
        comptime!(space.space().subspace(&[Q, S])),
        Attended {
            keys,
            queries,
            causal,
        },
    )
    .tile_in::<f32>(&space);
    for cube in space {
        let mut out_cube = out.at(&cube).planes_output::<f32>(&cube);
        for plane in cube {
            let (q_p, k_p, v_p) = (q.at(&plane), k.at(&plane), v.at(&plane));
            let bias_p = bias.at(&plane);
            // The plane's own keys, all of them or its share, and of those no block past the
            // attended keys.
            let origin = plane.origin(S);
            let reach = select(keys as usize > origin, keys as usize - origin, 0usize);
            let keys_walk = plane.walk();
            let blocks = reach.div_ceil(block_keys).min(keys_walk.total());
            let walk = keys_walk.range(0, blocks);
            let score = Tile::scratch::<f32>(&walk, comptime!(vec![Q, S]), StageStorage::Strided);
            let mut p = Tile::scratch::<E>(&walk, comptime!(vec![Q, S]), StageStorage::Strided);
            let acc = out
                .at(&plane)
                .accumulator::<f32>(&bias_p, &v_p, Instruction::Cmma, Semiring::SUM_PROD)
                .with_scratch(Scratch::OneTile);
            let mut softmax = OnlineSoftmax::<f32>::along(&score, S);
            // The query spans no key: staged once, before the walk, where the keys are staged.
            let mut q_s = q_p.stage_for(&walk, StageStorage::Strided, None);
            q_s.copy_from(&q_p);
            sync_plane();
            let mut stages = Stages::smem(&walk, &k_p, &v_p, StageStorage::Strided, 1usize, None);
            stages.pipelined(walk, |slot, stage| {
                let bias_s = bias_p.at(stage);
                slot.consume(|k_s, v_s| {
                    let logits =
                        score.accumulator::<f32>(&q_s, k_s, Instruction::Cmma, Semiring::SUM_PROD);
                    for fragment in stage.walk().routed(V, 0).unrolled() {
                        let mut cell = logits.at(&fragment);
                        cell.mma(&q_s.at(&fragment), &k_s.at(&fragment));
                    }
                    logits.drained_into(&score);
                    // The score's cells were stored in the fragments' layout; the softmax reads
                    // them slice by slice.
                    sync_plane();
                    let correction = softmax.step(&score, &bias_s, scale);
                    p.copy_cast_from(&score);
                    // The probabilities land before the fragments load them.
                    sync_plane();
                    acc.along(V).mul(&correction);
                    for fragment in stage.walk().routed(D, 0).unrolled() {
                        let mut cell = acc.at(&fragment);
                        cell.mma(&p.at(&fragment), &v_s.at(&fragment));
                    }
                });
            });
            acc.along(V).mul(&softmax.normalizer(&plane));
            out_cube.drain(&acc, &plane);
        }
        out_cube.write();
    }
}

/// [`plane_attention`] with the score held in manual-mma registers from the score's contraction to
/// the value's: the softmax reads the accumulator where its units hold it, and the probabilities,
/// cast in the same registers ([`Tile::cast`]), are the value's left factor. The leaf is one
/// instruction `m × n` along the queries and keys, and the value contracts the keys as deep as the
/// level above the leaf steps them: `n`, or a run of the score's fragments read as one `A`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn register_plane_attention<E: Float>(
    q: &TileArg<'_>,
    k: &TileArg<'_>,
    v: &TileArg<'_>,
    out: &TileArg<'_>,
    max: &mut Tensor<f32>,
    sum: &mut Tensor<f32>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] causal: bool,
    #[comptime] block_keys: usize,
    #[comptime] plane_rows: usize,
    #[comptime] ending: Ending,
    #[define(E)] _dtype: ElemType,
) {
    let instruction = comptime!(Instruction::Mma {
        io: MmaIo::default(),
        operands: MmaOperands::Plain,
    });
    let q = q.tile(&space);
    let k = k.tile(&space).within(S, 0, keys as usize);
    let v = v.tile(&space).within(S, 0, keys as usize);
    let out = out.tile(&space);
    let bias = Procedural::new::<f32, Attended>(
        comptime!(space.space().subspace(&[Q, S])),
        Attended {
            keys,
            queries,
            causal,
        },
    )
    .tile_in::<f32>(&space);
    for cube in space {
        let mut out_cube = out.at(&cube).planes_output::<f32>(&cube);
        for plane in cube {
            let (q_p, k_p, v_p) = (q.at(&plane), k.at(&plane), v.at(&plane));
            let bias_p = bias.at(&plane);
            let origin = plane.origin(S);
            let reach = select(keys as usize > origin, keys as usize - origin, 0usize);
            let keys_walk = plane.walk();
            let blocks = reach.div_ceil(block_keys).min(keys_walk.total());
            let walk = keys_walk.range(0, blocks);
            let acc =
                out.at(&plane)
                    .accumulator::<f32>(&bias_p, &v_p, instruction, Semiring::SUM_PROD);
            let mut q_s = q_p.stage_for(&walk, StageStorage::Strided, None);
            q_s.copy_from(&q_p);
            sync_plane();
            let logits =
                Tile::stage_accumulator::<f32>(&walk, comptime!(vec![Q, S]), &q_s, instruction);
            let mut softmax = OnlineSoftmax::<f32>::along(&logits, S);
            let mut stages = Stages::smem(&walk, &k_p, &v_p, StageStorage::Strided, 1usize, None);
            stages.pipelined(walk, |slot, stage| {
                let bias_s = bias_p.at(stage);
                slot.consume(|k_s, v_s| {
                    let mut logits = logits.clone();
                    logits.reset();
                    for fragment in stage.walk().routed(V, 0).unrolled() {
                        let mut cell = logits.at(&fragment);
                        cell.mma(&q_s.at(&fragment), &k_s.at(&fragment));
                    }
                    let correction = softmax.step(&logits, &bias_s, scale);
                    let p = logits.cast::<E>();
                    acc.along(V).mul(&correction);
                    for fragment in stage.walk().routed(D, 0).unrolled() {
                        let mut cell = acc.at(&fragment);
                        cell.mma(&p.at(&fragment), &v_s.at(&fragment));
                    }
                });
            });
            match comptime!(ending) {
                Ending::Normalized => acc.along(V).mul(&softmax.recip_l()),
                Ending::Undivided => {
                    // Each unit holds its own rows' state: written out row by row, then read by
                    // any unit.
                    let rows_walk = plane.walk().range(0, blocks);
                    let mut max_w =
                        Tile::scratch::<f32>(&rows_walk, comptime!(vec![Q]), StageStorage::Strided);
                    let mut sum_w =
                        Tile::scratch::<f32>(&rows_walk, comptime!(vec![Q]), StageStorage::Strided);
                    softmax.store_rows(&logits, &mut max_w, &mut sum_w);
                    sync_plane();
                    let (max_r, sum_r) =
                        (max_w.flat::<f32, Const<1>>(), sum_w.flat::<f32, Const<1>>());
                    let first = plane.origin(Q);
                    let mut r = UNIT_POS_PLANE as usize;
                    while r < plane_rows {
                        if first + r < queries as usize {
                            max[first + r] = max_r.read(r).extract(0usize);
                            sum[first + r] = sum_r.read(r).extract(0usize);
                        }
                        r += PLANE_DIM as usize;
                    }
                }
            }
            out_cube.drain(&acc, &plane);
        }
        out_cube.write();
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn unit_attention(
    q: &TileArg<'_>,
    k: &TileArg<'_>,
    v: &TileArg<'_>,
    out: &TileArg<'_>,
    max: &mut Tensor<f32>,
    sum: &mut Tensor<f32>,
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] causal: bool,
    #[comptime] block_keys: usize,
    #[comptime] rows: usize,
    #[comptime] ending: Ending,
) {
    let q = q.tile(&space);
    let k = k.tile(&space).within(S, 0, keys as usize);
    let v = v.tile(&space).within(S, 0, keys as usize);
    let out = out.tile(&space);
    let bias = Procedural::new::<f32, Attended>(
        comptime!(space.space().subspace(&[Q, S])),
        Attended {
            keys,
            queries,
            causal,
        },
    )
    .tile_in::<f32>(&space);
    let value = comptime!(space.space().extent(V));
    for cube in space {
        for unit in cube {
            let (q_u, k_u, v_u, out_u, bias_u) = (
                q.at(&unit),
                k.at(&unit),
                v.at(&unit),
                out.at(&unit),
                bias.at(&unit),
            );
            let mut acc = out_u.accumulator::<f32>(
                &bias_u,
                &v_u,
                comptime!(Instruction::Registers {
                    config: RegisterBlock::new(rows * value)
                }),
                Semiring::SUM_PROD,
            );
            let mut softmax = OnlineSoftmax::<f32>::along(&bias_u, S);
            for step in unit.walk().range(0, (keys as usize).div_ceil(block_keys)) {
                let (q_s, k_s, v_s, bias_s) = (
                    q_u.at(&step),
                    k_u.at(&step),
                    v_u.at(&step),
                    bias_u.at(&step),
                );
                let mut score = bias_s.accumulator::<f32>(
                    &q_s,
                    &k_s,
                    comptime!(Instruction::Registers {
                        config: RegisterBlock::new(rows * block_keys)
                    }),
                    Semiring::SUM_PROD,
                );
                score.mma(&q_s, &k_s);
                let correction = softmax.step(&score, &bias_s, scale);
                acc.along(V).mul(&correction);
                acc.mma(&score, &v_s);
            }
            match comptime!(ending) {
                Ending::Normalized => acc.along(V).mul(&softmax.recip_l()),
                Ending::Undivided => {
                    let state = softmax.state();
                    let first = unit.origin(Q);
                    #[unroll]
                    for r in 0..rows {
                        if first + r < queries as usize {
                            max[first + r] = state.max[r];
                            sum[first + r] = state.sum[r];
                        }
                    }
                }
            }
            acc.drained_into(&out_u);
        }
    }
}

/// [`plane_attention`] with the keys split across cubes: each cube walks its run of the blocks
/// with one plane, the cube level stepping the keys itself, then takes its turn at the rows'
/// carry. In its turn the plane merges its rows' state into `carried_max` and `carried_sum`,
/// rescales the carry and its own sum by the factors that returns, and adds its sum in.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn relayed_attention<E: Float>(
    q: &TileArg<'_>,
    k: &TileArg<'_>,
    v: &TileArg<'_>,
    out: &TileArg<'_>,
    carry: &TileArg<'_>,
    carried_max: &TileArg<'_>,
    carried_sum: &TileArg<'_>,
    turns: &[Atomic<u32>],
    keys: u32,
    queries: u32,
    scale: f32,
    space: Partitioning,
    #[comptime] causal: bool,
    #[comptime] block_keys: usize,
    #[define(E)] _dtype: ElemType,
) {
    let relay = carry.relay::<Const<1>>(&space, turns);
    // Uniform across the cube, so it leaves before any barrier.
    if !relay.has_turn() {
        terminate!();
    }
    let q = q.tile(&space);
    let k = k.tile(&space).within(S, 0, keys as usize);
    let v = v.tile(&space).within(S, 0, keys as usize);
    let mut out = out.tile(&space);
    let state_max = carried_max.tile(&space);
    let state_sum = carried_sum.tile(&space);
    let bias = Procedural::new::<f32, Attended>(
        comptime!(space.space().subspace(&[Q, S])),
        Attended {
            keys,
            queries,
            causal,
        },
    )
    .tile_in::<f32>(&space);
    // The cube level's walk is this cube's run of blocks, its planes directly below each block.
    let run = space.walk();
    let first = run.region(0);
    let origin = first.origin(S);
    let reach = select(keys as usize > origin, keys as usize - origin, 0usize);
    let blocks = reach.div_ceil(block_keys).min(run.total());
    let walk = run.range(0, blocks);
    let sum = relay.tile();
    let score = Tile::scratch::<f32>(&walk, comptime!(vec![Q, S]), StageStorage::Strided);
    let p = Tile::scratch::<E>(&walk, comptime!(vec![Q, S]), StageStorage::Strided);
    let acc = sum
        .accumulator::<f32>(&bias, &v, Instruction::Cmma, Semiring::SUM_PROD)
        .with_scratch(Scratch::OneTile);
    let mut softmax = OnlineSoftmax::<f32>::along(&score, S);
    // The query spans no key: its box is the first block's, staged once before the walk.
    let q_c = q.at(&first);
    let mut q_s = q_c.stage_for(&walk, StageStorage::Strided, None);
    q_s.copy_from(&q_c);
    sync_cube();
    let mut stages = Stages::smem(&walk, &k, &v, StageStorage::Strided, 1usize, None);
    stages.pipelined(walk, |slot, stage| {
        slot.consume(|k_s, v_s| {
            for plane in stage {
                let score_p = score.at(&plane);
                let logits = score_p.accumulator::<f32>(
                    &q_s.at(&plane),
                    &k_s.at(&plane),
                    Instruction::Cmma,
                    Semiring::SUM_PROD,
                );
                for fragment in plane.walk().routed(V, 0).unrolled() {
                    let mut cell = logits.at(&fragment);
                    cell.mma(&q_s.at(&fragment), &k_s.at(&fragment));
                }
                logits.drained_into(&score_p);
                sync_plane();
                let correction = softmax.step(&score_p, &bias.at(&plane), scale);
                let mut p_p = p.at(&plane);
                p_p.copy_cast_from(&score_p);
                sync_plane();
                acc.at(&plane).along(V).mul(&correction);
                for fragment in plane.walk().routed(D, 0).unrolled() {
                    let mut cell = acc.at(&fragment);
                    cell.mma(&p.at(&fragment), &v_s.at(&fragment));
                }
            }
        });
    });
    relay.take();
    let carried = relay.carried();
    let first_turn = relay.is_first();
    for plane in first.clone() {
        let factors = softmax.relayed(&relay, &mut state_max.at(&plane), &mut state_sum.at(&plane));
        acc.at(&plane).along(V).mul(&factors.sum);
        // The first turn's drain replaces the carry, which holds nothing yet.
        if !first_turn {
            carried.at(&plane).along(V).mul(&factors.carried);
        }
    }
    // Every row of the carry is rescaled before any plane's sum lands in it.
    sync_storage();
    for plane in first {
        acc.at(&plane).drained_into(&sum.at(&plane));
    }
    relay.pass(&mut out);
}

/// Which axis a cube's planes split: the queries, each plane owning its own rows, or the keys,
/// the planes merging what each computed of the same rows.
#[derive(Clone, Copy)]
enum PlanesAlong {
    Queries,
    Keys,
}

/// The walk over the blocks of keys and the planes above it: every block walked by each plane
/// owning its own queries, or a share of the `blocks` walked by each plane splitting the keys.
fn planes_over(steps: Levels, along: PlanesAlong, planes: usize, blocks: usize) -> Levels {
    match along {
        PlanesAlong::Queries => steps.walk_every(&[S]).planes(&[(Q, planes)]),
        PlanesAlong::Keys => steps.walk(&[(S, blocks / planes)]).planes(&[(S, planes)]),
    }
}

/// The problem's shape and how the kernel cuts it.
#[derive(Clone, Copy)]
struct Case {
    queries: usize,
    keys: usize,
    /// Keys attended, at most `keys`: the rest is poisoned.
    attended: usize,
    head: usize,
    value: usize,
    /// Fragment edge: the instruction is `edge × edge × edge`.
    edge: usize,
    /// Fragments a plane holds along the queries, and along a block of keys.
    rows: usize,
    block: usize,
    planes: usize,
    along: PlanesAlong,
    causal: bool,
}

fn run<E: Float + CubeElement>(case: Case) {
    let client: Client = cubecl::test_device().client();
    let e_ty = E::elem_type_native();
    let f32_ty = f32::elem_type_native();
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == e_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && [cfg.m, cfg.n, cfg.k] == [case.edge as u32; 3]
    });
    if !offered {
        ValidationResult::Skipped(format!(
            "no {0}x{0}x{0} {e_ty:?} fragment summing in f32",
            case.edge
        ))
        .enforce();
        return;
    }
    let Case {
        queries,
        keys,
        attended,
        head,
        value,
        edge,
        rows,
        block,
        planes,
        along,
        causal,
    } = case;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            planes_over(
                Levels::leaf(&[(Q, edge), (S, edge), (D, edge), (V, edge)]).walk(&[
                    (Q, rows),
                    (S, block),
                    (D, head / edge),
                    (V, value / edge),
                ]),
                along,
                planes,
                keys / (edge * block),
            )
            .cubes(&[Q])
            .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q_handle, k_handle, v_handle) =
        inputs::<E>(&client, queries, keys, attended, head, value, false);
    let out_handle = TestInput::builder(client.clone(), Shape::new([queries, value]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let scale = 1. / (head as f32).sqrt();

    let (q_tensor, k_tensor, v_tensor, out_tensor) = (
        q_handle.binding().into_tensor_arg(),
        k_handle.binding().into_tensor_arg(),
        v_handle.binding().into_tensor_arg(),
        out_handle.clone().binding().into_tensor_arg(),
    );
    plane_attention::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(q_tensor, e_ty, 1, TileSpec::direct(&[Q, D])),
        TileArgLaunch::new(k_tensor, e_ty, 1, TileSpec::direct(&[S, D])),
        TileArgLaunch::new(v_tensor, e_ty, 1, TileSpec::direct(&[S, V])),
        TileArgLaunch::new(out_tensor, f32_ty, 1, TileSpec::direct(&[Q, V])),
        attended as u32,
        queries as u32,
        scale,
        launcher.partitioning_arg(),
        causal,
        edge * block,
        e_ty,
    );
    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    check(
        &out, &q_data, &k_data, &v_data, queries, keys, attended, head, value, causal, false,
    );
}

/// A case through [`register_plane_attention`]: the leaf one `16 × 8` instruction along the
/// queries and keys, `16` deep along the head and `8` wide along the values, under a level that
/// steps the keys `run` leaves at a time. The score contracts `m16n8k16` on each leaf of a run,
/// and the value the run at once, `m16n8k8` for a run of one and `m16n8k16` for a run of two.
fn run_registers<E: Float + CubeElement>(case: Case, ending: Ending, run: usize) {
    let client: Client = cubecl::test_device().client();
    let e_ty = E::elem_type_native();
    let f32_ty = f32::elem_type_native();
    let offered = |k: u32| {
        client.properties().features.matmul.mma.iter().any(|cfg| {
            cfg.a_type == e_ty
                && cfg.b_type == e_ty
                && cfg.cd_type == f32_ty
                && [cfg.m, cfg.n, cfg.k] == [16, 8, k]
        })
    };
    let value_depth = 8 * run as u32;
    if !(offered(16) && offered(value_depth)) {
        ValidationResult::Skipped(format!(
            "no m16n8k16 and m16n8k{value_depth} {e_ty:?} instructions summing in f32"
        ))
        .enforce();
        return;
    }
    let Case {
        queries,
        keys,
        attended,
        head,
        value,
        rows,
        block,
        planes,
        causal,
        ..
    } = case;
    let (m, n, depth) = (16, 8, 16);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            planes_over(
                Levels::leaf(&[(Q, m), (S, n), (D, depth), (V, n)])
                    .walk(&[(S, run)])
                    .walk(&[
                        (Q, rows),
                        (S, block / run),
                        (D, head / depth),
                        (V, value / n),
                    ]),
                PlanesAlong::Queries,
                planes,
                keys / (n * block),
            )
            .cubes(&[Q])
            .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q_handle, k_handle, v_handle) =
        inputs::<E>(&client, queries, keys, attended, head, value, false);
    let out_handle = TestInput::builder(client.clone(), Shape::new([queries, value]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let state = || {
        TestInput::builder(client.clone(), Shape::new([queries]))
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data()
    };
    let (max_handle, sum_handle) = (state(), state());
    let scale = 1. / (head as f32).sqrt();
    register_plane_attention::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            q_handle.binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(&[Q, D]),
        ),
        TileArgLaunch::new(
            k_handle.binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(&[S, D]),
        ),
        TileArgLaunch::new(
            v_handle.binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(&[S, V]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[Q, V]),
        ),
        max_handle.clone().binding().into_tensor_arg(),
        sum_handle.clone().binding().into_tensor_arg(),
        attended as u32,
        queries as u32,
        scale,
        launcher.partitioning_arg(),
        causal,
        n * block,
        m * rows,
        ending,
        e_ty,
    );
    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    match ending {
        Ending::Normalized => check(
            &out, &q_data, &k_data, &v_data, queries, keys, attended, head, value, causal, false,
        ),
        Ending::Undivided => {
            let max = HostData::from_tensor_handle(&client, max_handle, HostDataType::F32);
            let sum = HostData::from_tensor_handle(&client, sum_handle, HostDataType::F32);
            // f16 operands and probabilities: a few ulps of each through two contractions.
            check_undivided(
                &out, &max, &sum, &q_data, &k_data, &v_data, queries, keys, attended, head, value,
                causal, false, 2e-2,
            );
        }
    }
}

/// A case whose keys `cubes` cubes split, each walking its run of blocks with one plane.
fn run_relayed<E: Float + CubeElement>(case: Case, cubes: usize) {
    let client: Client = cubecl::test_device().client();
    if !relays(&client) {
        return;
    }
    let e_ty = E::elem_type_native();
    let f32_ty = f32::elem_type_native();
    let offered = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == e_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && [cfg.m, cfg.n, cfg.k] == [case.edge as u32; 3]
    });
    if !offered {
        ValidationResult::Skipped(format!(
            "no {0}x{0}x{0} {e_ty:?} fragment summing in f32",
            case.edge
        ))
        .enforce();
        return;
    }
    let Case {
        queries,
        keys,
        attended,
        head,
        value,
        edge,
        rows,
        block,
        causal,
        ..
    } = case;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            Levels::leaf(&[(Q, edge), (S, edge), (D, edge), (V, edge)])
                .walk(&[(Q, rows), (S, block), (D, head / edge), (V, value / edge)])
                .planes(&[(Q, 1)])
                .cubes(&[Q, S])
                .across(S, cubes)
                .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q_handle, k_handle, v_handle) =
        inputs::<E>(&client, queries, keys, attended, head, value, false);
    let buffer = |shape: Shape| {
        TestInput::builder(client.clone(), shape)
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data()
    };
    let out_handle = buffer(Shape::new([queries, value]));
    let (carry, carried_max, carried_sum) = (
        buffer(Shape::new([queries, value])),
        buffer(Shape::new([queries])),
        buffer(Shape::new([queries])),
    );
    let counters = launcher.partitioning().relay_counters();
    let turns = client.create_from_slice(u32::as_bytes(&vec![0u32; counters]));
    let tensor = |handle: cubecl::std::tensor::TensorHandle| handle.binding().into_tensor_arg();
    relayed_attention::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(tensor(q_handle), e_ty, 1, TileSpec::direct(&[Q, D])),
        TileArgLaunch::new(tensor(k_handle), e_ty, 1, TileSpec::direct(&[S, D])),
        TileArgLaunch::new(tensor(v_handle), e_ty, 1, TileSpec::direct(&[S, V])),
        TileArgLaunch::new(
            tensor(out_handle.clone()),
            f32_ty,
            1,
            TileSpec::direct(&[Q, V]),
        ),
        TileArgLaunch::new(tensor(carry), f32_ty, 1, TileSpec::direct(&[Q, V])),
        TileArgLaunch::new(tensor(carried_max), f32_ty, 1, TileSpec::direct(&[Q])),
        TileArgLaunch::new(tensor(carried_sum), f32_ty, 1, TileSpec::direct(&[Q])),
        unsafe { BufferArg::from_raw_parts(turns, counters) },
        attended as u32,
        queries as u32,
        1. / (head as f32).sqrt(),
        launcher.partitioning_arg(),
        causal,
        edge * block,
        e_ty,
    );
    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    check(
        &out, &q_data, &k_data, &v_data, queries, keys, attended, head, value, causal, false,
    );
}

/// The query, keys and values of a case and their host copies. Every key past the attended ones
/// is NaN, so reading one shows; the keys are stored `[head, keys]` where `transposed_keys`.
#[allow(clippy::type_complexity)]
fn inputs<E: Float + CubeElement>(
    client: &Client,
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    transposed_keys: bool,
) -> (
    HostData,
    HostData,
    HostData,
    cubecl::std::tensor::TensorHandle,
    cubecl::std::tensor::TensorHandle,
    cubecl::std::tensor::TensorHandle,
) {
    let e_ty = E::elem_type_native();
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
    let key_cell = |j: usize, d: usize| match j < attended {
        true => wobble(j * head + d, 2),
        false => f32::NAN,
    };
    let key_values: Vec<f32> = match transposed_keys {
        true => (0..head * keys)
            .map(|i| key_cell(i % keys, i / keys))
            .collect(),
        false => (0..keys * head)
            .map(|i| key_cell(i / head, i % head))
            .collect(),
    };
    let key_shape = match transposed_keys {
        true => Shape::new([head, keys]),
        false => Shape::new([keys, head]),
    };
    let (q_handle, q_data) = TestInput::builder(client.clone(), Shape::new([queries, head]))
        .dtype(e_ty)
        .custom((0..queries * head).map(|i| wobble(i, 1)).collect())
        .generate_with_f32_host_data();
    let (k_handle, k_data) = TestInput::builder(client.clone(), key_shape)
        .dtype(e_ty)
        .custom(key_values)
        .generate_with_f32_host_data();
    let (v_handle, v_data) = TestInput::builder(client.clone(), Shape::new([keys, value]))
        .dtype(e_ty)
        .custom(
            (0..keys * value)
                .map(|i| match i / value < attended {
                    true => wobble(i, 3),
                    false => f32::NAN,
                })
                .collect(),
        )
        .generate_with_f32_host_data();
    (q_data, k_data, v_data, q_handle, k_handle, v_handle)
}

/// Row `i` of the attention computed on the host in `f64`: the max of its scaled logits, the
/// sum of their `exp(logit − max)`, and its output, normalized. A query sees the attended keys,
/// and under `causal` none past its own position aligned at the bottom right; a row that sees
/// none has a zero sum and a zero output.
#[allow(clippy::too_many_arguments)]
fn reference_row(
    i: usize,
    q_data: &HostData,
    k_data: &HostData,
    v_data: &HostData,
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    causal: bool,
    transposed_keys: bool,
) -> (f64, f64, Vec<f64>) {
    let scale = 1. / (head as f64).sqrt();
    let key = |j: usize, d: usize| match transposed_keys {
        true => k_data.get_f32(&[d, j]),
        false => k_data.get_f32(&[j, d]),
    };
    let seen = |j: usize| j < attended && (!causal || j + queries <= i + attended);
    let logits: Vec<(usize, f64)> = (0..keys)
        .filter(|&j| seen(j))
        .map(|j| {
            let dot: f64 = (0..head)
                .map(|d| q_data.get_f32(&[i, d]) as f64 * key(j, d) as f64)
                .sum();
            (j, dot * scale)
        })
        .collect();
    let max = logits.iter().fold(f64::NEG_INFINITY, |m, &(_, s)| m.max(s));
    let sum: f64 = logits.iter().map(|&(_, s)| (s - max).exp()).sum();
    let out = (0..value)
        .map(|c| match logits.is_empty() {
            true => 0.,
            false => {
                logits
                    .iter()
                    .map(|&(j, s)| (s - max).exp() * v_data.get_f32(&[j, c]) as f64)
                    .sum::<f64>()
                    / sum
            }
        })
        .collect();
    (max, sum, out)
}

/// `out` against the attention computed on the host ([`reference_row`]).
#[allow(clippy::too_many_arguments)]
fn check(
    out: &HostData,
    q_data: &HostData,
    k_data: &HostData,
    v_data: &HostData,
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    causal: bool,
    transposed_keys: bool,
) {
    for i in 0..queries {
        let (_, _, row) = reference_row(
            i,
            q_data,
            k_data,
            v_data,
            queries,
            keys,
            attended,
            head,
            value,
            causal,
            transposed_keys,
        );
        for (c, &expected) in row.iter().enumerate() {
            let actual = out.get_f32(&[i, c]) as f64;
            // f16 operands and probabilities: a few ulps of each through two contractions.
            let tolerance = 2e-2 * (1. + expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "out[{i}, {c}] = {actual}, expected {expected}"
            );
        }
    }
}

/// An undivided ending against the host's rows ([`reference_row`]): each row's `max` and `sum` as
/// the state held them, and its `out` the normalized output times that sum, each within
/// `tolerance` of it relative to one plus its size. A row that saw no key holds a zero sum and a
/// zero output.
#[allow(clippy::too_many_arguments)]
fn check_undivided(
    out: &HostData,
    max: &HostData,
    sum: &HostData,
    q_data: &HostData,
    k_data: &HostData,
    v_data: &HostData,
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    causal: bool,
    transposed_keys: bool,
    tolerance: f64,
) {
    let close =
        |actual: f64, expected: f64| (actual - expected).abs() <= tolerance * (1. + expected.abs());
    for i in 0..queries {
        let (row_max, row_sum, row) = reference_row(
            i,
            q_data,
            k_data,
            v_data,
            queries,
            keys,
            attended,
            head,
            value,
            causal,
            transposed_keys,
        );
        let (held_max, held_sum) = (max.get_f32(&[i]) as f64, sum.get_f32(&[i]) as f64);
        assert!(
            close(held_sum, row_sum),
            "sum[{i}] = {held_sum}, expected {row_sum}"
        );
        if row_sum > 0. {
            assert!(
                close(held_max, row_max),
                "max[{i}] = {held_max}, expected {row_max}"
            );
        }
        for (c, &normalized) in row.iter().enumerate() {
            let (actual, expected) = (out.get_f32(&[i, c]) as f64, normalized * row_sum);
            assert!(
                close(actual, expected),
                "out[{i}, {c}] = {actual}, expected {expected}"
            );
        }
    }
}

/// How a holder leaves its rows: normalized, or undivided beside the softmax state it read,
/// which the host divides by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Ending {
    Normalized,
    Undivided,
}

/// A unit-owned case: `rows` queries a unit, `units` units a cube, `block` keys a step, and how
/// each unit ends.
#[derive(Clone, Copy)]
struct UnitCase {
    queries: usize,
    keys: usize,
    attended: usize,
    head: usize,
    value: usize,
    rows: usize,
    units: usize,
    block: usize,
    causal: bool,
    ending: Ending,
}

fn run_units<E: Float + CubeElement>(case: UnitCase) {
    let client: Client = cubecl::test_device().client();
    let e_ty = E::elem_type_native();
    let f32_ty = f32::elem_type_native();
    let UnitCase {
        queries,
        keys,
        attended,
        head,
        value,
        rows,
        units,
        block,
        causal,
        ending,
    } = case;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(Q, queries), (S, keys), (D, head), (V, value)]),
            Levels::leaf(&[(Q, rows), (S, block), (D, head), (V, value)])
                .walk_every(&[S])
                .units_distributed(Q, units)
                .cubes(&[Q])
                .build(),
        ),
        Form::Static,
    );
    let (q_data, k_data, v_data, q_handle, k_handle, v_handle) =
        inputs::<E>(&client, queries, keys, attended, head, value, true);
    let out_handle = TestInput::builder(client.clone(), Shape::new([queries, value]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let state = || {
        TestInput::builder(client.clone(), Shape::new([queries]))
            .dtype(f32_ty)
            .zeros()
            .generate_without_host_data()
    };
    let (max_handle, sum_handle) = (state(), state());
    let scale = 1. / (head as f32).sqrt();
    unit_attention::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            q_handle.binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(&[Q, D]),
        ),
        // The keys are stored transposed, so a step's rhs runs along the keys it scores.
        TileArgLaunch::new(
            k_handle.binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(&[D, S]),
        ),
        TileArgLaunch::new(
            v_handle.binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(&[S, V]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[Q, V]),
        ),
        max_handle.clone().binding().into_tensor_arg(),
        sum_handle.clone().binding().into_tensor_arg(),
        attended as u32,
        queries as u32,
        scale,
        launcher.partitioning_arg(),
        causal,
        block,
        rows,
        ending,
    );
    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    match ending {
        Ending::Normalized => check(
            &out, &q_data, &k_data, &v_data, queries, keys, attended, head, value, causal, true,
        ),
        Ending::Undivided => {
            let max = HostData::from_tensor_handle(&client, max_handle, HostDataType::F32);
            let sum = HostData::from_tensor_handle(&client, sum_handle, HostDataType::F32);
            // f32 operands summed in f32: a few ulps through one contraction and the
            // exponentials. The keys are stored transposed.
            check_undivided(
                &out, &max, &sum, &q_data, &k_data, &v_data, queries, keys, attended, head, value,
                causal, true, 1e-4,
            );
        }
    }
}

const PREFILL: Case = Case {
    queries: 32,
    keys: 64,
    attended: 64,
    head: 16,
    value: 16,
    edge: 8,
    rows: 2,
    block: 2,
    planes: 2,
    along: PlanesAlong::Queries,
    causal: false,
};

#[test]
fn plane_attention_matches_the_reference() {
    run::<half::f16>(PREFILL);
}

#[test]
fn plane_attention_reads_nothing_past_the_attended_keys() {
    run::<half::f16>(Case {
        attended: 40,
        ..PREFILL
    });
}

#[test]
fn plane_attention_masks_causally_from_the_bottom_right() {
    run::<half::f16>(Case {
        attended: 50,
        causal: true,
        ..PREFILL
    });
}

#[test]
fn plane_attention_in_f32() {
    run::<f32>(Case {
        value: 24,
        ..PREFILL
    });
}

/// A plane's rows of a prefill held in manual-mma registers: two fragments along the queries, a
/// stage of four along the keys.
const REGISTER_PREFILL: Case = Case {
    queries: 64,
    keys: 128,
    attended: 128,
    head: 32,
    value: 32,
    edge: 0,
    rows: 2,
    block: 4,
    planes: 2,
    along: PlanesAlong::Queries,
    causal: false,
};

#[test]
fn register_plane_attention_matches_the_reference() {
    run_registers::<half::f16>(REGISTER_PREFILL, Ending::Normalized, 1);
}

#[test]
fn register_plane_attention_reads_nothing_past_the_attended_keys() {
    run_registers::<half::f16>(
        Case {
            attended: 90,
            ..REGISTER_PREFILL
        },
        Ending::Normalized,
        1,
    );
}

#[test]
fn register_plane_attention_masks_causally_from_the_bottom_right() {
    run_registers::<half::f16>(
        Case {
            attended: 100,
            causal: true,
            ..REGISTER_PREFILL
        },
        Ending::Normalized,
        1,
    );
}

/// Each unit holding only its own rows' state, written out row by row
/// ([`OnlineSoftmax::store_rows`]) beside the undivided sum: past the attended keys and causally,
/// so a masked cell weighs nothing in either.
#[test]
fn register_plane_attention_hands_its_undivided_state_by_rows() {
    run_registers::<half::f16>(
        Case {
            attended: 100,
            causal: true,
            ..REGISTER_PREFILL
        },
        Ending::Undivided,
        1,
    );
}

/// The value contracting two of the score's fragments at once, `m16n8k16` reading both as its
/// `A`: causally and past the attended keys, so the masked cells of a run weigh nothing.
#[test]
fn register_plane_attention_contracts_a_run_of_two_score_fragments() {
    run_registers::<half::f16>(REGISTER_PREFILL, Ending::Normalized, 2);
    run_registers::<half::f16>(
        Case {
            attended: 100,
            causal: true,
            ..REGISTER_PREFILL
        },
        Ending::Normalized,
        2,
    );
}

/// Planes splitting the keys of the same queries: two, then four, each walking a share.
const SPLIT_KEYS: Case = Case {
    queries: 16,
    keys: 128,
    attended: 128,
    planes: 2,
    along: PlanesAlong::Keys,
    ..PREFILL
};

#[test]
fn plane_attention_split_keys_matches_the_reference() {
    run::<half::f16>(SPLIT_KEYS);
    run::<half::f16>(Case {
        planes: 4,
        ..SPLIT_KEYS
    });
}

/// A share past the attended keys walks nothing, and its plane still meets the others: every
/// slice it holds is masked, so it adds nothing to the merged sum.
#[test]
fn plane_attention_split_keys_reads_nothing_past_the_attended_keys() {
    run::<half::f16>(Case {
        attended: 40,
        planes: 4,
        ..SPLIT_KEYS
    });
}

#[test]
fn plane_attention_split_keys_masks_causally_from_the_bottom_right() {
    run::<half::f16>(Case {
        attended: 100,
        causal: true,
        ..SPLIT_KEYS
    });
}

/// Cubes splitting the keys of the same queries: two, then four, each walking a run of the
/// blocks and taking its turn at the rows.
#[test]
fn relayed_attention_matches_the_reference() {
    run_relayed::<half::f16>(SPLIT_KEYS, 2);
    run_relayed::<half::f16>(SPLIT_KEYS, 4);
}

/// A run past the attended keys walks nothing, and its cube still takes its turn: every slice it
/// holds is masked, so it rescales nothing away and adds nothing.
#[test]
fn relayed_attention_reads_nothing_past_the_attended_keys() {
    run_relayed::<half::f16>(
        Case {
            attended: 40,
            ..SPLIT_KEYS
        },
        4,
    );
}

#[test]
fn relayed_attention_masks_causally_from_the_bottom_right() {
    run_relayed::<half::f16>(
        Case {
            attended: 100,
            causal: true,
            ..SPLIT_KEYS
        },
        4,
    );
}

const UNIT: UnitCase = UnitCase {
    queries: 24,
    keys: 40,
    attended: 40,
    head: 16,
    value: 8,
    rows: 2,
    units: 4,
    block: 4,
    causal: false,
    ending: Ending::Normalized,
};

#[test]
fn unit_attention_matches_the_reference() {
    run_units::<f32>(UNIT);
}

#[test]
fn unit_attention_reads_nothing_past_the_attended_keys() {
    run_units::<f32>(UnitCase {
        attended: 27,
        ..UNIT
    });
}

#[test]
fn unit_attention_masks_causally_from_the_bottom_right() {
    run_units::<f32>(UnitCase {
        attended: 30,
        causal: true,
        ..UNIT
    });
}

#[test]
fn unit_attention_in_f16() {
    run_units::<half::f16>(UNIT);
}

/// The state read off as it stands, its sum left undivided: past the attended keys and
/// causally, so a masked cell weighs nothing in either.
#[test]
fn unit_attention_hands_its_undivided_state() {
    run_units::<f32>(UnitCase {
        attended: 30,
        causal: true,
        ending: Ending::Undivided,
        ..UNIT
    });
}
