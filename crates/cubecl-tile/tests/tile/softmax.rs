//! Numerics tests for the attention softmax leaf.
//!
//! The kernel below is a miniature of the future Fold body: walk S blocks, stage each score
//! block into an smem tile, run `Tile::softmax`, rescale a running accumulator by the returned
//! correction (a `val_dim` = 1 value matmul), then normalize by `l` and store `lse = m + ln(l)`.
//!
//! Checked against direct (non-online) host math, including exact zeros and exact -inf lse on
//! fully-masked rows.

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use crate::tile::uncut;
use cubecl::features::Plane;
use cubecl::std::tensor::layout::CoordsDyn;
use cubecl::{client::Client, prelude::*, zspace::Shape};
use cubecl_tile::{
    Axis, Level, Partitioning, Space, StageStorage, Tile, TileArg, TileArgLaunch, TileSpec,
    ops::softmax::{MaskProbe, RowState},
};

const Q: Axis = Axis(0);
const S: Axis = Axis(1);

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn softmax_walk_kernel(
    score_in: &TileArg<'_>, // {Q: rows, S: total_cols} raw scores
    mask: &TileArg<'_>,     // {Q, S} boolean, nonzero = masked
    values: &Tensor<f32>,   // [total_cols]
    out: &mut Tensor<f32>,  // [rows]
    lse: &mut Tensor<f32>,  // [rows]
    scale: f32,
    bound_s: u32,
    space: Partitioning,
    #[comptime] block_space: Space, // {Q: rows, S: block cols}
    #[comptime] units: usize,
    #[comptime] plane_units: usize,
    #[comptime] causal: bool,
    #[comptime] materialized: bool,
    #[comptime] num_blocks: usize,
) {
    let score_gmem = score_in.tile(&space);
    let mask_tile = mask.tile(&space);
    let mut score = Tile::smem::<f32>(block_space.clone(), 1usize, StageStorage::Strided, units);
    let mut p = Tile::smem::<f32>(block_space.clone(), 1usize, StageStorage::Strided, units);

    let rows = comptime!(block_space.extent(Q));
    let cols = comptime!(block_space.extent(S));
    // One worker per row-slice, where a worker is a unit or a whole plane. A plane is handed its
    // own window of the tiles (the leaf indexes no planes); a unit reads its rows off the whole
    // tile. `units == 1` is the unit arm, so everything below reads the same either way.
    let planes = comptime!(units / plane_units);
    let rpu = comptime!(rows.div_ceil(planes));
    let kept_space = comptime!(Space::new(&[(Q, rpu)]));
    let mut state = match comptime!(plane_units > 1) {
        true => RowState::<f32>::over_plane(kept_space, plane_units),
        false => RowState::<f32>::new(comptime!(Space::new(&[(Q, rows)])), units),
    };
    let plane_unit = UNIT_POS_X as usize % plane_units;
    let worker = UNIT_POS_X as usize / plane_units;
    let rows_level = comptime!(Level::every(&[(Q, rpu), (S, cols)]));
    let mut acc = Array::<f32>::new(rpu);
    for ri in 0..rpu {
        acc[ri] = 0.0;
    }

    for blk in 0..num_blocks {
        // Stage the block: each worker fills its own rows and, inside them,
        // each unit the columns the leaf will read back — so no cell crosses a
        // unit and no sync is owed on either arm.
        let gmem = score_gmem.view::<f32, Const<1>>();
        let mut smem = score.view_mut::<f32, Const<1>>();
        for ri in 0..rpu {
            let r = worker * rpu + ri;
            if r < rows {
                let mut c = plane_unit;
                while c < cols {
                    let mut src = CoordsDyn::new();
                    src.push(r as u32);
                    src.push((blk * cols + c) as u32);
                    let mut dst = CoordsDyn::new();
                    dst.push(r as u32);
                    dst.push(c as u32);
                    smem.write(dst, gmem.read(src));
                    c += plane_units;
                }
            }
        }

        let probe = MaskProbe {
            origin_q: 0,
            row_origin: worker * rpu,
            origin_s: blk * cols,
            bound_q: rows,
            bound_s: bound_s as usize,
            q_rows: rows,
            causal,
            materialized,
        };
        let corr = if comptime!(plane_units > 1) {
            let mut score_w = score.at(&score.over(&rows_level).region(worker));
            let mut p_w = p.at(&p.over(&rows_level).region(worker));
            score_w.softmax::<f32>(&mut p_w, &mut state, &probe, &mask_tile, scale)
        } else {
            let probe = MaskProbe {
                origin_q: 0,
                row_origin: 0,
                origin_s: blk * cols,
                bound_q: rows,
                bound_s: bound_s as usize,
                q_rows: rows,
                causal,
                materialized,
            };
            score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale)
        };

        // The block update `O = corr·O + P·V`, on a scalar accumulator. Each unit sums the columns
        // it owns and the plane closes it, which is the value matmul's own shape; at one unit the
        // reduction is the identity.
        let p_view = p.view::<f32, Const<1>>();
        for ri in 0..rpu {
            let r = worker * rpu + ri;
            if r < rows {
                acc[ri] *= corr[ri];
                let mut part = 0f32;
                let mut c = plane_unit;
                while c < cols {
                    let mut pos = CoordsDyn::new();
                    pos.push(r as u32);
                    pos.push(c as u32);
                    part += p_view.read(pos).extract(0usize) * values[blk * cols + c];
                    c += plane_units;
                }
                acc[ri] += match comptime!(plane_units > 1) {
                    true => plane_sum(part),
                    false => part,
                };
            }
        }
    }

    // The state is worker-uniform, so one unit of each writes its rows.
    for ri in 0..rpu {
        let r = worker * rpu + ri;
        if r < rows && plane_unit == 0 {
            out[r] = acc[ri] * state.recip_l(ri);
            lse[r] = state.lse(ri);
        }
    }
}

/// Launch the walk kernel at unit ownership and check out/lse against direct
/// host math.
fn run(
    shape: (usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    mask_fn: Option<fn(usize, usize) -> bool>,
) {
    run_at(1, shape, bound_s, causal, mask_fn);
}

/// [`run`] at plane ownership, at the width this device commits to.
///
/// Skipped where it commits to none: a plane reduction over a fabricated or
/// ranged width folds the wrong units and is silently wrong, which is why the
/// arm takes its width from the caller rather than reading `PLANE_DIM`.
fn run_planar(
    shape: (usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    mask_fn: Option<fn(usize, usize) -> bool>,
) {
    let client: Client = cubecl::test_device().client();
    let hardware = &client.properties().hardware;
    let exact = hardware.plane_size_min == hardware.plane_size_max;
    if !client.properties().features.plane.contains(Plane::Ops) || !exact {
        ValidationResult::Skipped("device commits to no exact plane width".to_string()).enforce();
        return;
    }
    run_at(
        hardware.plane_size_min as usize,
        shape,
        bound_s,
        causal,
        mask_fn,
    );
}

/// The body both arms share: `units` units make one worker.
fn run_at(
    plane_units: usize,
    (units, rows, cols, num_blocks): (usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
    mask_fn: Option<fn(usize, usize) -> bool>,
) {
    let client: Client = cubecl::test_device().client();
    let total_cols = cols * num_blocks;
    let scale = 0.125f32;

    // The CPU runtime caps units per cube at core count; ownership only needs
    // a consistent unit count, so clamp (rows_per_unit grows to compensate).
    let units = units.min(client.properties().hardware.max_units_per_cube as usize);
    // A worker is a whole plane, so the cube holds a whole number of them.
    let units = units.next_multiple_of(plane_units);

    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    // Deterministic host-built data so device and host math see identical bits.
    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
    let (score_handle, score_data) =
        TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
            .dtype(f32_ty)
            .custom((0..rows * total_cols).map(|i| wobble(i, 1)).collect())
            .generate_with_f32_host_data();

    let mask_values: Vec<f32> = (0..rows * total_cols)
        .map(|lin| {
            let (i, j) = (lin / total_cols, lin % total_cols);
            mask_fn.map(|f| f(i, j)).unwrap_or(false) as u32 as f32
        })
        .collect();
    let mask_handle = TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
        .dtype(u32_ty)
        .custom(mask_values)
        .generate_without_host_data();

    let (values_handle, values_data) = TestInput::builder(client.clone(), Shape::new([total_cols]))
        .dtype(f32_ty)
        .custom((0..total_cols).map(|i| wobble(i, 2) / 2.).collect())
        .generate_with_f32_host_data();

    let out_handle = TestInput::builder(client.clone(), Shape::new([rows]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let lse_handle = TestInput::builder(client.clone(), Shape::new([rows]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let gmem_space = Space::new(&[(Q, rows), (S, total_cols)]);
    let block_space = Space::new(&[(Q, rows), (S, cols)]);

    softmax_walk_kernel::launch(
        &client,
        CubeCount::new_single(),
        // Explicit x = units so UNIT_POS_X is the owner index on every
        // backend (CubeDim::new packs by plane size: y-major on CPU).
        CubeDim::new_2d(units as u32, 1),
        TileArgLaunch::new(
            score_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[Q, S]),
        ),
        TileArgLaunch::new(
            mask_handle.clone().binding().into_tensor_arg(),
            u32_ty,
            1,
            TileSpec::direct(&[Q, S]),
        ),
        values_handle.clone().binding().into_tensor_arg(),
        out_handle.clone().binding().into_tensor_arg(),
        lse_handle.clone().binding().into_tensor_arg(),
        scale,
        bound_s as u32,
        uncut(&client, &gmem_space, &gmem_space).partitioning_arg(),
        block_space,
        units,
        plane_units,
        causal,
        mask_fn.is_some(),
        num_blocks,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let lse = HostData::from_tensor_handle(&client, lse_handle, HostDataType::F32);

    // Guard against the cubecl-cpu smem-aliases-input bug: the kernel never
    // writes its inputs, so any change means Shared aliased a live binding.
    let score_after = HostData::from_tensor_handle(&client, score_handle, HostDataType::F32);
    let values_after = HostData::from_tensor_handle(&client, values_handle, HostDataType::F32);
    let _keep_alive = mask_handle;
    for i in 0..rows {
        for j in 0..total_cols {
            assert_eq!(
                score_after.get_f32(&[i, j]),
                score_data.get_f32(&[i, j]),
                "input score corrupted at ({i},{j}): smem aliased an input binding"
            );
        }
    }
    for j in 0..total_cols {
        assert_eq!(
            values_after.get_f32(&[j]),
            values_data.get_f32(&[j]),
            "input values corrupted at {j}: smem aliased an input binding"
        );
    }

    for i in 0..rows {
        // Direct math: gather unmasked scaled scores, then one-pass softmax.
        let mut scores = Vec::new();
        for j in 0..total_cols {
            let oob = j >= bound_s;
            let causal_masked = causal && j > i;
            let mat_masked = mask_fn.map(|f| f(i, j)).unwrap_or(false);
            if !(oob || causal_masked || mat_masked) {
                scores.push((
                    score_data.get_f32(&[i, j]) * scale,
                    values_data.get_f32(&[j]),
                ));
            }
        }

        let got_out = out.get_f32(&[i]);
        let got_lse = lse.get_f32(&[i]);

        if scores.is_empty() {
            assert_eq!(got_out, 0., "fully-masked row {i}: out must be exactly 0");
            assert_eq!(
                got_lse,
                f32::NEG_INFINITY,
                "fully-masked row {i}: lse must be exactly -inf"
            );
            continue;
        }

        let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
        let expected_lse = m + l.ln();
        let expected_out: f32 = scores.iter().map(|(s, v)| (s - m).exp() / l * v).sum();

        assert!(
            (got_out - expected_out).abs() <= 1e-5 * expected_out.abs().max(1.),
            "row {i}: out {got_out} vs direct {expected_out}"
        );
        assert!(
            (got_lse - expected_lse).abs() <= 1e-5 * expected_lse.abs().max(1.),
            "row {i}: lse {got_lse} vs direct {expected_lse}"
        );
    }
}

const V: Axis = Axis(2);

/// The fold with a shared-memory `{Q, V}` accumulator, the shape the attention routine takes:
/// row owners publish their correction in a rank-1 factors tile ([`store_rows`](cubecl_tile::Tile))
/// and the whole cube rescales the accumulator by it ([`scale_rows`](cubecl_tile::Tile)).
///
/// The value accumulate runs under a *different* ownership (cyclic) than the softmax rows: the
/// cross-unit handoff the smem path exists for.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn softmax_smem_acc_kernel(
    score_in: &TileArg<'_>, // {Q: rows, S: total_cols} raw scores
    mask: &TileArg<'_>,     // unused (materialized = false)
    values: &Tensor<f32>,   // [total_cols, val_dim] row-major
    out: &mut Tensor<f32>,  // [rows, val_dim] row-major
    lse: &mut Tensor<f32>,  // [rows]
    scale: f32,
    bound_s: u32,
    space: Partitioning,
    #[comptime] block_space: Space, // {Q: rows, S: block cols}
    #[comptime] units: usize,
    #[comptime] causal: bool,
    #[comptime] num_blocks: usize,
    #[comptime] val_dim: usize,
) {
    let score_gmem = score_in.tile(&space);
    let mask_tile = mask.tile(&space);
    let mut score = Tile::smem::<f32>(block_space.clone(), 1usize, StageStorage::Strided, units);
    let mut p = Tile::smem::<f32>(block_space.clone(), 1usize, StageStorage::Strided, units);

    let rows = comptime!(block_space.extent(Q));
    let cols = comptime!(block_space.extent(S));
    let kept_space = comptime!(Space::new(&[(Q, rows)]));
    let mut state = RowState::<f32>::new(kept_space.clone(), units);
    let share = comptime!(state.share);
    let rpu = comptime!(share.rows());

    let mut factors = Tile::smem::<f32>(kept_space, 1usize, StageStorage::Strided, units);
    let acc_space = comptime!(Space::new(&[(Q, rows), (V, val_dim)]));
    let mut acc = Tile::smem::<f32>(acc_space, 1usize, StageStorage::Strided, units);
    acc.zero();

    for blk in 0..num_blocks {
        // Stage the block: each unit fills its own rows.
        let gmem = score_gmem.view::<f32, Const<1>>();
        let mut smem = score.view_mut::<f32, Const<1>>();
        for ri in 0..rpu {
            let r = UNIT_POS_X as usize * rpu + ri;
            if r < rows {
                for c in 0..cols {
                    let mut src = CoordsDyn::new();
                    src.push(r as u32);
                    src.push((blk * cols + c) as u32);
                    let mut dst = CoordsDyn::new();
                    dst.push(r as u32);
                    dst.push(c as u32);
                    smem.write(dst, gmem.read(src));
                }
            }
        }

        let probe = MaskProbe {
            origin_q: 0,
            row_origin: 0,
            origin_s: blk * cols,
            bound_q: rows,
            bound_s: bound_s as usize,
            q_rows: rows,
            causal,
            materialized: false,
        };
        let corr = score.softmax::<f32>(&mut p, &mut state, &probe, &mask_tile, scale);
        factors.store_rows(&corr, &state);
        sync_cube();

        // `O = corr·O`, every cell exactly once, whoever owns the rows.
        acc.scale_rows(&factors);
        sync_cube();

        // `O += P·V_blk`, cyclic ownership: deliberately not the softmax rows'.
        let p_view = p.view::<f32, Const<1>>();
        let mut acc_view = acc.view_mut::<f32, Const<1>>();
        let total = comptime!(rows * val_dim);
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let r = i / val_dim;
            let v = i % val_dim;
            let mut pos = CoordsDyn::new();
            pos.push(r as u32);
            pos.push(v as u32);
            let mut cell = acc_view.read(pos.clone());
            for c in 0..cols {
                let mut ppos = CoordsDyn::new();
                ppos.push(r as u32);
                ppos.push(c as u32);
                let prob = p_view.read(ppos).extract(0usize);
                cell += Vector::cast_from(prob * values[(blk * cols + c) * val_dim + v]);
            }
            acc_view.write(pos, cell);
            i += workers;
        }
        sync_cube();
    }

    // Epilogue: owners publish `1/l`, the cube normalizes, everyone drains.
    let mut recip = Array::<f32>::new(rpu);
    for ri in 0..rpu {
        recip[ri] = state.recip_l(ri);
    }
    factors.store_rows(&recip, &state);
    sync_cube();
    acc.scale_rows(&factors);
    sync_cube();

    let acc_view = acc.view::<f32, Const<1>>();
    let total = comptime!(rows * val_dim);
    let workers = CUBE_DIM as usize;
    let mut i = UNIT_POS as usize;
    while i < total {
        let mut pos = CoordsDyn::new();
        pos.push((i / val_dim) as u32);
        pos.push((i % val_dim) as u32);
        out[i] = acc_view.read(pos).extract(0usize);
        i += workers;
    }
    for ri in 0..rpu {
        let r = UNIT_POS_X as usize * rpu + ri;
        if r < rows {
            lse[r] = state.lse(ri);
        }
    }
}

/// Launch the smem-accumulator kernel and check out/lse against direct host math.
fn run_smem_acc(
    (units, rows, cols, num_blocks, val_dim): (usize, usize, usize, usize, usize),
    bound_s: usize,
    causal: bool,
) {
    let client: Client = cubecl::test_device().client();
    let total_cols = cols * num_blocks;
    let scale = 0.125f32;
    let units = units.min(client.properties().hardware.max_units_per_cube as usize);

    let f32_ty = f32::elem_type_native();
    let u32_ty = u32::elem_type_native();

    let wobble =
        |i: usize, salt: usize| ((i * 2654435761 + salt * 40503) % 2048) as f32 / 512. - 2.;
    let (score_handle, score_data) =
        TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
            .dtype(f32_ty)
            .custom((0..rows * total_cols).map(|i| wobble(i, 1)).collect())
            .generate_with_f32_host_data();
    let mask_handle = TestInput::builder(client.clone(), Shape::new([rows, total_cols]))
        .dtype(u32_ty)
        .zeros()
        .generate_without_host_data();
    let (values_handle, values_data) =
        TestInput::builder(client.clone(), Shape::new([total_cols, val_dim]))
            .dtype(f32_ty)
            .custom(
                (0..total_cols * val_dim)
                    .map(|i| wobble(i, 2) / 2.)
                    .collect(),
            )
            .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), Shape::new([rows, val_dim]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();
    let lse_handle = TestInput::builder(client.clone(), Shape::new([rows]))
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let gmem_space = Space::new(&[(Q, rows), (S, total_cols)]);
    let block_space = Space::new(&[(Q, rows), (S, cols)]);

    softmax_smem_acc_kernel::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_2d(units as u32, 1),
        TileArgLaunch::new(
            score_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[Q, S]),
        ),
        TileArgLaunch::new(
            mask_handle.clone().binding().into_tensor_arg(),
            u32_ty,
            1,
            TileSpec::direct(&[Q, S]),
        ),
        values_handle.clone().binding().into_tensor_arg(),
        out_handle.clone().binding().into_tensor_arg(),
        lse_handle.clone().binding().into_tensor_arg(),
        scale,
        bound_s as u32,
        uncut(&client, &gmem_space, &gmem_space).partitioning_arg(),
        block_space,
        units,
        causal,
        num_blocks,
        val_dim,
    );

    let out = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let lse = HostData::from_tensor_handle(&client, lse_handle, HostDataType::F32);

    for i in 0..rows {
        let mut scores = Vec::new();
        for j in 0..total_cols {
            let oob = j >= bound_s;
            let causal_masked = causal && j > i;
            if !(oob || causal_masked) {
                scores.push((score_data.get_f32(&[i, j]) * scale, j));
            }
        }

        let got_lse = lse.get_f32(&[i]);
        if scores.is_empty() {
            assert_eq!(
                got_lse,
                f32::NEG_INFINITY,
                "fully-masked row {i}: lse must be exactly -inf"
            );
            for v in 0..val_dim {
                assert_eq!(
                    out.get_f32(&[i, v]),
                    0.,
                    "fully-masked row {i}: out must be exactly 0"
                );
            }
            continue;
        }

        let m = scores.iter().fold(f32::NEG_INFINITY, |m, (s, _)| m.max(*s));
        let l: f32 = scores.iter().map(|(s, _)| (s - m).exp()).sum();
        let expected_lse = m + l.ln();
        assert!(
            (got_lse - expected_lse).abs() <= 1e-5 * expected_lse.abs().max(1.),
            "row {i}: lse {got_lse} vs direct {expected_lse}"
        );
        for v in 0..val_dim {
            let expected: f32 = scores
                .iter()
                .map(|(s, j)| (s - m).exp() / l * values_data.get_f32(&[*j, v]))
                .sum();
            let got = out.get_f32(&[i, v]);
            assert!(
                (got - expected).abs() <= 1e-5 * expected.abs().max(1.),
                "row ({i},{v}): out {got} vs direct {expected}"
            );
        }
    }
}

/// The decode shape: a handful of GQA rows, a full plane, deep-ish walk.
#[test]
fn smem_acc_decode_rows() {
    run_smem_acc((32, 4, 16, 4, 8), 60, false);
}

/// Causal with several rows per unit and a `val_dim` wider than the plane.
#[test]
fn smem_acc_causal_multi_rows() {
    run_smem_acc((8, 16, 8, 3, 40), 20, true);
}

/// Rows 0..2 fully masked, row 2 first-half, row 3 second-half (running max
/// starts at -inf / all-masked late blocks), scattered elsewhere.
fn crafted_mask(i: usize, j: usize) -> bool {
    i < 2 || (i == 2 && j < 16) || (i == 3 && j >= 16) || (i + j).is_multiple_of(5)
}

/// One unit owns every row: the unit/CPU shape (`SoftmaxKind::Direct` twin).
#[test]
fn direct_unit_masked() {
    run((1, 8, 8, 4), 27, false, Some(crafted_mask));
}

/// One row per unit, a cube wide.
#[test]
fn one_row_per_unit_masked() {
    run((32, 32, 16, 2), 30, false, Some(crafted_mask));
}

/// **Plane ownership**: a plane owns the row, its units split the columns and
/// the reduction closes in the hardware. Same rows, same mask, same answer as
/// [`one_row_per_unit_masked`] — the arm changes who computes, never what.
///
/// The mask is what makes it a real test of the split: a unit whose columns
/// are all masked contributes the identity, and a max seeded on every unit
/// must not survive as one row's answer.
#[test]
fn plane_masked() {
    run_planar((32, 32, 16, 2), 30, false, Some(crafted_mask));
}

/// Plane ownership with more rows than planes, so a plane folds several rows
/// in sequence and its `rpp` slots must stay independent.
#[test]
fn plane_multi_rows_masked_causal() {
    run_planar((64, 8, 16, 3), 40, true, Some(crafted_mask));
}

/// Plane ownership on a block narrower than the plane: most units contribute
/// nothing to the row, so the reduction's identities are load-bearing — a sum
/// seeded wrong or a max left at zero shows up here and nowhere else.
#[test]
fn plane_block_narrower_than_the_plane() {
    run_planar((32, 4, 8, 2), 11, false, None);
}

/// Plane ownership over a fully-masked prefix: `l` stays exactly zero, the drain divides by it
/// through the masked guard, and `lse` is -inf. The plane arm reaches that guard through a
/// reduction where the unit arm does not, so it is worth its own case.
#[test]
fn plane_fully_masked_rows() {
    run_planar((32, 16, 8, 2), 0, false, None);
}

/// Several rows per unit (rows/units = 2).
#[test]
fn multi_rows_per_unit_masked_causal() {
    run((4, 8, 8, 2), 14, true, Some(crafted_mask));
}

#[test]
fn causal_only() {
    run((16, 16, 8, 2), 16, true, None);
}

/// Causal with bound past the diagonal: early rows see one unmasked block
/// then fully-masked blocks; late rows the reverse.
#[test]
fn causal_odd_bound() {
    run((8, 8, 4, 3), 10, true, None);
}
