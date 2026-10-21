//! The attention fold end-to-end on tiles under the hardware instruction: the general contraction
//! on a plane-resident accumulator, interleaved with the softmax step and its row rescale, the
//! walk owned by the kernel. The software leaves and the split and streamed folds belong to the
//! client downstream, and are tested there.
//!
//! GQA rides the axes: `q` stacks its group over the same query block (group-major rows), `k`/`v`
//! omit the group axis, and the probe's `q_rows` maps rows back to query positions for the causal
//! predicate.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Axis, Instruction, Level, Levels, Partitioning, Scratch, Semiring, Space, StageStorage, Tile,
    TileArg, TileArgLaunch, TileSpec,
    ops::softmax::{MaskProbe, RowState},
};

const G: Axis = Axis(0); // GQA group member
const QP: Axis = Axis(1); // query position
const S: Axis = Axis(2); // key/value position (the reduced axis)
const D: Axis = Axis(3); // head dim (contracted by the score matmul)
const V: Axis = Axis(4); // value dim
// Local labels for the kernel-allocated smem tiles.
const R: Axis = Axis(5); // score rows = G × QP, group-major
const C: Axis = Axis(6); // score cols = one S block

/// The same fold with both matmuls on tensor cores: the score and mix leaves are the tensor-core
/// ones, so a plane owns whole fragments where a unit owned columns, and the softmax between them
/// is unchanged scalar shared-memory work.
///
/// The flow llama.cpp's `fa.metal` runs: Q@K into fragments stored to shared memory, the online
/// softmax on plain floats with the running total rescaled where it lies, then P@V folding onto an
/// accumulator fragment loaded back from smem. Nothing persists in a fragment across a barrier.
///
/// Here each plane owns `rows / planes` query rows through all three phases: the score fills the
/// plane's window of the smem score tile, the softmax runs on that window, and the mix folds into
/// an accumulator kept in fragments across the walk. The per-block K/V fills are the only barriers.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn attention_fold_cmma_kernel<E: Float>(
    q: &TileArg<'_>,    // {QP, D}
    k: &TileArg<'_>,    // {S, D}
    v: &TileArg<'_>,    // {S, V}
    mask: &TileArg<'_>, // 1-cell dummy (materialized = false)
    out: &TileArg<'_>,  // {QP, V}
    scale: f32,
    bound: u32,
    space: Partitioning,
    #[comptime] blocks: Level,
    #[comptime] causal: bool,
    #[comptime] block: usize,
    #[comptime] frag: usize,
    #[comptime] planes: usize,
    #[comptime] score_vec: usize,
    #[comptime] plane_units: usize,
    #[define(E)] _dtype: ElemType,
) {
    let q = q.tile(&space);
    let k = k.tile(&space);
    let v = v.tile(&space);
    let mask_tile = mask.tile(&space);
    let out = out.tile(&space);

    let rows = comptime!(q.space().extent(QP));
    let d = comptime!(q.space().extent(D));
    let val_dim = comptime!(v.space().extent(V));
    let rows_p = comptime!(rows / planes);
    let (rm, cn, vn, ks) = comptime!((rows_p / frag, block / frag, val_dim / frag, d / frag));

    let mut q_s = Tile::smem::<E>(
        comptime!(Space::new(&[(QP, rows), (D, d)])),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    q_s.copy_from(&q);
    let score = Tile::smem::<f32>(
        comptime!(Space::new(&[(QP, rows), (S, block)])),
        score_vec,
        StageStorage::Strided,
        0usize,
    );
    // The block's window of K and of V, staged by the cube and read by every plane.
    let k_walk = k.over(&blocks);
    let k_probe = k.at(&k_walk.region(0usize));
    let v_probe = v.at(&k_walk.region(0usize));
    let mut k_stage = Tile::shared::<E>(comptime!(k_probe.space()), StageStorage::Strided);
    let mut v_stage = Tile::shared::<E>(comptime!(v_probe.space()), StageStorage::Strided);
    let bound_s = bound as usize;
    sync_cube();

    for plane in space.over(&comptime!(
        Levels::leaf(&[(QP, rows_p)])
            .planes(&[(QP, planes)])
            .level()
    )) {
        let q_w = q_s.at(&plane);
        let mut score_w = score.at(&plane);
        let out_w = out.at(&plane);
        // The stages carry no query axis, so the plane's window is the whole block, one level
        // down like everything else it reads.
        let k_w = k_stage.at(&plane);
        let v_w = v_stage.at(&plane);
        let row_origin = plane.coord(QP) * rows_p;

        let mut state =
            RowState::<f32>::over_plane(comptime!(Space::new(&[(QP, rows_p)])), plane_units);
        // The accumulators' grids are stated as levels: the plane level above, and below it the
        // fragment grid each is cut to, which is what sizes the fragments and the scratch.
        let plane_level = comptime!(
            Levels::leaf(&[(QP, rows_p)])
                .planes(&[(QP, planes)])
                .level()
        );
        let out_g = out_w.with_levels(
            1usize,
            comptime!(vec![
                plane_level.clone(),
                Levels::leaf(&[(QP, frag), (V, frag), (S, frag)])
                    .walk(&[(QP, rm), (V, vn), (S, cn)])
                    .level(),
            ]),
        );
        let score_g = score_w.with_levels(
            1usize,
            comptime!(vec![
                plane_level.clone(),
                Levels::leaf(&[(QP, frag), (S, frag), (D, frag)])
                    .walk(&[(QP, rm), (S, cn), (D, ks)])
                    .level(),
            ]),
        );
        let mut acc = out_g
            .accumulator::<f32>(&score_w, &v_w, Instruction::Cmma, Semiring::SUM_PROD)
            .with_scratch(Scratch::OneTile);
        // The fragment grids of every operand, cells in row-major order.
        let acc_cells = out_w.over(&comptime!(Level::every(&[(QP, frag), (V, frag)])));
        let p_cells = score_w.over(&comptime!(Level::every(&[(QP, frag), (S, frag)])));
        let q_cells = q_w.over(&comptime!(Level::every(&[(QP, frag), (D, frag)])));
        let k_cells = k_w.over(&comptime!(Level::every(&[(S, frag), (D, frag)])));
        let v_cells = v_w.over(&comptime!(Level::every(&[(S, frag), (V, frag)])));

        // The probe states the masking once and the walk takes it as its bound: a block every row
        // masks throughout is never stepped to, rather than contracted and discarded by the
        // softmax. Its rows still mask per element, for the last block's tail and its diagonal.
        let probe = MaskProbe {
            origin_q: 0,
            row_origin,
            origin_s: 0,
            bound_q: rows.runtime(),
            bound_s,
            q_rows: rows,
            causal,
            materialized: false,
        };

        for region in k.over(&blocks).range(0, probe.blocks(block)) {
            let s0 = region.coord(S) * block;
            let cols_bound = max(bound_s, s0) - s0;
            // Every plane is through the previous block before its stages are overwritten.
            sync_cube();
            k_stage.copy_from(&k.at(&region));
            v_stage.copy_from(&v.at(&region));
            sync_cube();

            // The score: `q · kᵀ`, the keys' window read col-major by the leaf.
            let s = score_g.accumulator::<f32>(&q_w, &k_w, Instruction::Cmma, Semiring::SUM_PROD);
            #[unroll]
            for si in 0..ks {
                #[unroll]
                for mi in 0..rm {
                    #[unroll]
                    for ni in 0..cn {
                        let mut cell = s.at(&p_cells.region(comptime!(mi * cn + ni)));
                        cell.mma(
                            &q_w.at(&q_cells.region(comptime!(mi * ks + si))),
                            &k_w.at(&k_cells.region(comptime!(ni * ks + si))),
                        );
                    }
                }
            }
            #[unroll]
            for i in 0..comptime!(rm * cn) {
                let cell = p_cells.region(i);
                let mut window = score_w.at(&cell);
                window.copy_cast_from(&s.at(&cell));
            }
            sync_plane();

            let probe = probe.step_s(s0);
            let corr = score_w.softmax_in_place(&mut state, &probe, &mask_tile, scale);
            acc.rescale_rows(&corr, &state);
            sync_plane();

            // The mix: `p · v`, steps at or past the prefix skipped so stale cache never rides
            // a zero probability.
            #[unroll]
            for si in 0..cn {
                if si * frag < cols_bound {
                    #[unroll]
                    for mi in 0..rm {
                        #[unroll]
                        for ni in 0..vn {
                            let mut cell = acc.at(&acc_cells.region(comptime!(mi * vn + ni)));
                            cell.mma(
                                &score_w.at(&p_cells.region(comptime!(mi * cn + si))),
                                &v_w.at(&v_cells.region(comptime!(si * vn + ni))),
                            );
                        }
                    }
                }
            }
            sync_plane();
        }

        let mut recip = Array::<f32>::new(rows_p);
        #[unroll]
        for ri in 0..rows_p {
            recip[ri] = state.recip_l(ri);
        }
        acc.rescale_rows(&recip, &state);
        sync_plane();
        #[unroll]
        for i in 0..comptime!(rm * vn) {
            let cell = acc_cells.region(i);
            let mut window = out_w.at(&cell);
            window.copy_cast_from(&acc.at(&cell));
        }
    }
}

/// Launch the hardware fold: `planes` planes of `units`, each owning `rows / planes` rows, over
/// `block`-wide steps of `s_total` keys with `frag` fragments, and check against direct host
/// math.
#[allow(clippy::too_many_arguments)]
fn run_cmma<E: Float + CubeElement>(
    (rows, s_total, block, d, val_dim, frag): (usize, usize, usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    spanned: bool,
    planes: usize,
    score_vec: usize,
) {
    let client: Client = cubecl::test_device().client();
    let hw = &client.properties().hardware;
    if hw.plane_size_min != hw.plane_size_max {
        ValidationResult::Skipped(
            "a plane-owned fold needs a plane width the device commits to".into(),
        )
        .enforce();
        return;
    }
    let plane_units = hw.plane_size_min as usize;
    if plane_units * planes > hw.max_units_per_cube as usize {
        ValidationResult::Skipped(format!(
            "{planes} planes of {plane_units} do not fit one cube here"
        ))
        .enforce();
        return;
    }
    let f32_ty = f32::elem_type_native();
    let e_ty = E::elem_type_native();
    let supported = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == e_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && (cfg.m as usize, cfg.n as usize, cfg.k as usize) == (frag, frag, frag)
    });
    let mix_supported = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == f32_ty
            && cfg.b_type == e_ty
            && cfg.cd_type == f32_ty
            && (cfg.m as usize, cfg.n as usize, cfg.k as usize) == (frag, frag, frag)
    });
    if !supported || !mix_supported {
        ValidationResult::Skipped(format!(
            "device has no {frag}x{frag}x{frag} {e_ty:?} cmma fragment accumulating at f32"
        ))
        .enforce();
        return;
    }
    let scale = 1. / (d as f32).sqrt();

    let u32_ty = u32::elem_type_native();
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;

    let (q_handle, q_data) = TestInput::builder(client.clone(), Shape::new([rows, d]))
        .dtype(e_ty)
        .custom((0..rows * d).map(|i| wobble(i, 1)).collect())
        .generate_with_f32_host_data();
    // The spanned arm binds the same values under one more dim: same cells, one axis above the
    // two the leaf contracts.
    let kv_shape = |rows: usize, cols: usize| {
        if spanned {
            Shape::new([1, rows, cols])
        } else {
            Shape::new([rows, cols])
        }
    };
    let (k_handle, k_data) = TestInput::builder(client.clone(), kv_shape(s_total, d))
        .dtype(e_ty)
        .custom((0..s_total * d).map(|i| wobble(i, 2)).collect())
        .generate_with_f32_host_data();
    let (v_handle, v_data) = TestInput::builder(client.clone(), kv_shape(s_total, val_dim))
        .dtype(e_ty)
        .custom((0..s_total * val_dim).map(|i| wobble(i, 3) / 2.).collect())
        .generate_with_f32_host_data();
    let mask_handle = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(u32_ty)
        .zeros()
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([rows, val_dim]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // The launch walks `S` in blocks and nothing else; the planes' cut on `QP` is the kernel's.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (G, 1),
                (QP, rows),
                (S, s_total),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ]),
            Levels::leaf(&[
                (G, 1),
                (QP, rows),
                (S, block),
                (D, d),
                (V, val_dim),
                (R, 1),
                (C, 1),
            ])
            .walk_every(&[G, QP, S, D, V, R, C])
            .build(),
        ),
        Form::Static,
    );
    let (k_axes, v_axes): (&[Axis], &[Axis]) = if spanned {
        (&[G, S, D], &[G, S, V])
    } else {
        (&[S, D], &[S, V])
    };

    attention_fold_cmma_kernel::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(plane_units as u32, planes as u32),
        TileArgLaunch::new(
            q_handle.clone().binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(&[QP, D]),
        ),
        TileArgLaunch::new(
            k_handle.clone().binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(k_axes),
        ),
        TileArgLaunch::new(
            v_handle.clone().binding().into_tensor_arg(),
            e_ty,
            1,
            TileSpec::direct(v_axes),
        ),
        TileArgLaunch::new(
            mask_handle.clone().binding().into_tensor_arg(),
            u32_ty,
            1,
            TileSpec::direct(&[R, C]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[QP, V]),
        ),
        scale,
        bound_s as u32,
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        causal,
        block,
        frag,
        planes,
        score_vec,
        plane_units,
        e_ty,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);

    for r in 0..rows {
        let mut scores = Vec::new();
        for j in 0..s_total {
            let masked = j >= bound_s || (causal && j > r);
            if !masked {
                let dot: f32 = (0..d)
                    .map(|p| {
                        let key = if spanned {
                            k_data.get_f32(&[0, j, p])
                        } else {
                            k_data.get_f32(&[j, p])
                        };
                        q_data.get_f32(&[r, p]) * key
                    })
                    .sum();
                scores.push((dot * scale, j));
            }
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
        for vi in 0..val_dim {
            let expected: f32 = scores
                .iter()
                .map(|(s, j)| {
                    let value = if spanned {
                        v_data.get_f32(&[0, *j, vi])
                    } else {
                        v_data.get_f32(&[*j, vi])
                    };
                    (s - m).exp() / l * value
                })
                .sum();
            let got = out.get_f32(&[r, vi]);
            assert!(
                (got - expected).abs() <= 1e-4 * expected.abs().max(1.),
                "row {r} v {vi}: out {got} vs direct {expected}"
            );
        }
    }
}

/// One fragment: 8 rows on one plane, one head-dim step, one block of 8 keys. A leaf reading
/// the wrong window comes out as zeros or as one wrong block.
#[test]
fn fold_cmma_single_fragment() {
    run_cmma::<f32>((8, 16, 8, 8, 8, 8), 16, false, false, 1, 1);
}

/// A grid on one plane: 16 rows, two fragments tall, the head dim and the block two steps deep.
/// A step dropped leaves half a dot product; a fragment stored to the wrong cell a wrong row.
#[test]
fn fold_cmma_fragment_grid() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 32, false, false, 1, 1);
}

/// The attended prefix ends inside a block: the mask probe owns the tail, score fragments that
/// straddle the bound are contracted whole, and only whole value steps past it are skipped.
#[test]
fn fold_cmma_causal_ragged_bound() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, false, 1, 1);
}

/// K and V carry an axis above the two the contraction reads, spanned at one position.
#[test]
fn fold_cmma_spanned_leading_axis() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, true, 1, 1);
}

/// Two planes, each owning eight of sixteen rows through the score, the softmax and the mix, with
/// the accumulator resident in fragments and rescaled by the plane's own corrections. A plane
/// reading another's rows, a misapplied correction, or a misplaced drain shows up as a wrong row.
#[test]
fn fold_cmma_two_planes() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, true, 2, 1);
}

/// Four planes, one fragment of rows each, the accumulator several fragments wide.
#[test]
fn fold_cmma_four_planes() {
    run_cmma::<f32>((32, 32, 16, 16, 32, 8), 32, true, false, 4, 1);
}

/// Half operands: `f16` queries, keys and values, scores and the running total at `f32`, the mix
/// contracting `f32` probabilities against `f16` values.
#[test]
fn fold_cmma_half_operands() {
    run_cmma::<half::f16>((16, 32, 16, 16, 16, 8), 24, true, true, 2, 1);
}

/// A lined score tile: the fragments store and load it through its element slice, and the
/// softmax reads a row a line at a time.
#[test]
fn fold_cmma_lined_scores() {
    run_cmma::<f32>((16, 32, 16, 16, 16, 8), 24, true, false, 2, 4);
}

/// What the probe's bound decides, measured rather than inferred from a result the per-element
/// masking would make right either way: the walk's own step count, written out.
#[cube(launch)]
fn visited_blocks_kernel(
    k: &TileArg<'_>,
    out: &mut Tensor<f32>,
    bound: u32,
    space: Partitioning,
    #[comptime] blocks: Level,
    #[comptime] block: usize,
    #[comptime] q_rows: usize,
    #[comptime] causal: bool,
) {
    let k = k.tile(&space);
    let probe = MaskProbe {
        origin_q: 0,
        row_origin: 0,
        origin_s: 0,
        bound_q: comptime!(q_rows).runtime(),
        bound_s: bound as usize,
        q_rows,
        causal,
        materialized: false,
    };
    let mut visited = 0u32;
    for _region in k.over(&blocks).range(0, probe.blocks(block)) {
        visited += 1;
    }
    out[0] = f32::cast_from(visited);
}

/// `S` total, in blocks of `BLOCK`: four blocks to skip out of.
const VISIT_S: usize = 16;
const VISIT_BLOCK: usize = 4;
const VISIT_D: usize = 4;

fn visited_blocks(bound_s: usize, q_rows: usize, causal: bool) -> usize {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(S, VISIT_S), (D, VISIT_D)]),
            Levels::leaf(&[(S, VISIT_BLOCK)]).walk_every(&[S]).build(),
        ),
        Form::Static,
    );

    let (k_handle, _) = TestInput::builder(client.clone(), Shape::new([VISIT_S, VISIT_D]))
        .dtype(f32_ty)
        .custom(vec![0.0; VISIT_S * VISIT_D])
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([1]))
        .dtype(f32_ty)
        .custom(vec![999.0])
        .generate_without_host_data();

    visited_blocks_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            k_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[S, D]),
        ),
        out_handle.clone().binding().into_tensor_arg(),
        bound_s as u32,
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        VISIT_BLOCK,
        q_rows,
        causal,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32).get_f32(&[0]) as usize
}

/// A block every row masks throughout is never stepped to. Without the bound each case walks all
/// four blocks and the softmax throws the extra scores away, which is what this measures the
/// absence of.
#[test]
fn the_probe_bounds_the_walk_to_the_blocks_its_rows_can_read() {
    // Nothing masked: the whole axis, so the bound costs nothing.
    assert_eq!(visited_blocks(VISIT_S, VISIT_S, false), 4);
    // Causal with four query rows: only the first block holds a key any row may see.
    assert_eq!(visited_blocks(VISIT_S, 4, true), 1);
    // Causal with eight: two blocks.
    assert_eq!(visited_blocks(VISIT_S, 8, true), 2);
    // A ragged bound alone, no causality: six keys is two blocks, the second partial.
    assert_eq!(visited_blocks(6, VISIT_S, false), 2);
    // Both, the tighter one deciding.
    assert_eq!(visited_blocks(6, 4, true), 1);
    // Nothing to read: the walk takes no steps at all.
    assert_eq!(visited_blocks(0, VISIT_S, true), 0);
}
