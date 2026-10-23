//! A separable procedural weight contracted over three axes: the rank the abstraction carries is
//! the recipe's, not the microkernel's.
//!
//! `out[row, col] = Σ_{k0, k1, k2} (∏_f factor_f(k_f, row)) · in[k0, k1, k2, col]`. The same
//! recipe is run twice, once stating its factorization and once opaque, so the cell-major walk is
//! checked against the general one as well as against the host.
#![allow(non_snake_case)]

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{ir::ElemType, prelude::*, zspace::shape};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::procedural::AffineCoordinate;
use cubecl_tile::procedural::DivGuard;
use cubecl_tile::procedural::Factors;
use cubecl_tile::procedural::Procedural;
use cubecl_tile::procedural::Sum;
use cubecl_tile::procedural::TapSupport;
use cubecl_tile::procedural::affine_along;
use cubecl_tile::procedural::sum_of;
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};

const ROW: Axis = Axis(0);
const COL: Axis = Axis(1);
const TAP: [Axis; 3] = [Axis(2), Axis(3), Axis(4)];

/// The software instruction every leaf here runs under: a 16-cell budget, no edge split, no unit
/// fan-out.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

const ROWS: usize = 2;
const COLS: usize = 3;
/// Deliberately unequal, so a factor read at the wrong offset lands on the wrong tap count.
const TAPS: [usize; 3] = [2, 3, 2];

/// `offset + coefficient * tap + row_coefficient * row`, per factor. Only the first factor reads
/// the output row, which is the shape a resampling weight takes: a factor may read a free axis,
/// it may only not read another factor's contracted axis.
const OFFSET: [f32; 3] = [1.0, 2.0, 3.0];
const COEFFICIENT: [f32; 3] = [1.0, 3.0, 5.0];
const ROW_COEFFICIENT: [f32; 3] = [2.0, 0.0, 0.0];

/// Every factor is the same type, which is what a [`Factors`] holds: one filter family
/// applied along each axis, at that axis's own coordinates.
type Factor<E> = Sum<AffineCoordinate<E>, AffineCoordinate<E>>;
type Weights<E> = Factors<Factor<E>>;

fn factor_value(f: usize, tap: usize, row: usize) -> f32 {
    OFFSET[f] + COEFFICIENT[f] * tap as f32 + ROW_COEFFICIENT[f] * row as f32
}

#[cube]
fn factor<E: Float>(#[comptime] f: usize) -> Factor<E> {
    sum_of(
        affine_along(
            comptime!(TAP[f]),
            E::new(comptime!(f64::from(OFFSET[f]))),
            E::new(comptime!(f64::from(COEFFICIENT[f]))),
        ),
        affine_along(
            ROW,
            E::new(0.0),
            E::new(comptime!(f64::from(ROW_COEFFICIENT[f]))),
        ),
    )
}

#[cube]
fn weights<E: Float>() -> Weights<E> {
    let mut factors = Sequence::new();
    #[unroll]
    for f in 0..comptime!(TAP.len()) {
        factors.push(factor::<E>(f));
    }
    Factors::new(factors)
}

#[cube(launch)]
fn separable_kernel<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    #[comptime] separable: bool,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weight_axes = comptime!([&[ROW], TAP.as_slice()].concat());
    let weights = if comptime!(separable) {
        Procedural::separable::<E, Weights<E>>(
            comptime!(space.space().subspace(&weight_axes)),
            weights::<E>(),
        )
        .tile::<E>()
    } else {
        Procedural::new::<E, Weights<E>>(
            comptime!(space.space().subspace(&weight_axes)),
            weights::<E>(),
        )
        .tile::<E>()
    };

    let output = output.tile(&space);
    for region in space.over(&level) {
        let mut out = output
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        out.mm(&weights.at(&region), &input.at(&region));
    }
}

/// [`separable_kernel`] with the input staged into shared memory, served in `width`-wide lines
/// where one is stated (the operand is scalar, the stage pads it out to whole lines).
#[cube(launch)]
fn separable_kernel_staged<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    #[comptime] width: Option<usize>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weight_axes = comptime!([&[ROW], TAP.as_slice()].concat());
    let weights = Procedural::separable::<E, Weights<E>>(
        comptime!(space.space().subspace(&weight_axes)),
        weights::<E>(),
    )
    .tile::<E>();

    let output = output.tile(&space);
    let walk = space.over(&level);
    let mut stages =
        Stages::smem_single_at(&walk, &input, StageStorage::Strided, width, 1usize, None);
    stages.pipelined(walk, |slot, region| {
        let mut out = output
            .at(region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        let weights = weights.at(region);
        slot.consume(|input| {
            out.mm(&weights, input);
        });
    });
}

/// Small integers, so the accumulation is exact in `f32`.
fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 7) as f32) - 2.0).collect()
}

fn reference(input: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; ROWS * COLS];
    for row in 0..ROWS {
        for col in 0..COLS {
            let mut acc = 0.0f32;
            for k0 in 0..TAPS[0] {
                for k1 in 0..TAPS[1] {
                    for k2 in 0..TAPS[2] {
                        let weight = factor_value(0, k0, row)
                            * factor_value(1, k1, row)
                            * factor_value(2, k2, row);
                        let at = ((k0 * TAPS[1] + k1) * TAPS[2] + k2) * COLS + col;
                        acc += weight * input[at];
                    }
                }
            }
            out[row * COLS + col] = acc;
        }
    }
    out
}

fn run(separable: bool) -> (HostData, Vec<f32>) {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let in_shape = shape![TAPS[0], TAPS[1], TAPS[2], COLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (ROW, ROWS),
                (COL, COLS),
                (TAP[0], TAPS[0]),
                (TAP[1], TAPS[1]),
                (TAP[2], TAPS[2]),
            ]),
            Levels::leaf(&[
                (ROW, ROWS),
                (COL, COLS),
                (TAP[0], TAPS[0]),
                (TAP[1], TAPS[1]),
                (TAP[2], TAPS[2]),
            ])
            .walk_every(&[ROW, COL, TAP[0], TAP[1], TAP[2]])
            .build(),
        ),
        Form::Static,
    );

    separable_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            in_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[TAP[0], TAP[1], TAP[2], COL]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        separable,
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    (
        HostData::from_tensor_handle(&client, out_handle, HostDataType::F32),
        in_data,
    )
}

fn check(separable: bool) {
    let (got, input) = run(separable);
    let want = reference(&input);
    for row in 0..ROWS {
        for col in 0..COLS {
            let have = got.get_f32(&[row, col]);
            let want = want[row * COLS + col];
            assert!(
                (have - want).abs() < 1e-4,
                "separable {separable}: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

/// Three factors, walked one axis at a time: `2 + 3 + 2` evaluations per cell rather than
/// `2 * 3 * 2`, and the same values.
#[test]
fn a_rank_three_separable_product_contracts_factor_by_factor() {
    check(true);
}

/// The same recipe with its factorization withheld, contracted the general way. A recipe is the
/// product of its factors, so both paths must agree.
#[test]
fn an_opaque_product_contracts_to_the_same_values() {
    check(false);
}

#[test]
fn a_separable_lhs_contracts_a_padded_staged_rhs() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let in_shape = shape![TAPS[0], TAPS[1], TAPS[2], COLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (ROW, ROWS),
                (COL, COLS),
                (TAP[0], TAPS[0]),
                (TAP[1], TAPS[1]),
                (TAP[2], TAPS[2]),
            ]),
            Levels::leaf(&[
                (ROW, ROWS),
                (COL, COLS),
                (TAP[0], TAPS[0]),
                (TAP[1], TAPS[1]),
                (TAP[2], TAPS[2]),
            ])
            .walk_every(&[ROW, COL, TAP[0], TAP[1], TAP[2]])
            .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::direct(&[TAP[0], TAP[1], TAP[2], COL]);

    separable_kernel_staged::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        Some(4),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let want = reference(&in_data);
    for row in 0..ROWS {
        for col in 0..COLS {
            let have = got.get_f32(&[row, col]);
            let want = want[row * COLS + col];
            assert!(
                (have - want).abs() < 1e-4,
                "separable padded staged: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

// ---- separable lhs against a quantized rhs ---------------------------------

// ---- separable lhs against a resampling rhs --------------------------------

/// The rhs's one gathered physical axis: `⌊(3·row + 2·tap) / 2⌋`, which is `⌊3·row / 2⌋ + tap`.
///
/// Both halves of the split the separable schedule runs on are load-bearing here. `row` stays
/// inside the floor, so it has to be anchored; `tap` has a coefficient the divisor factors out, so
/// it steps by `1` on that anchor. This pins the hand-written step against evaluating the whole map per tap.
const ROW_NUM: usize = 3;
const RESAMPLE: usize = 2;
const RTAPS: usize = 2;
const RROWS: usize = 4;
const RCOLS: usize = 3;

fn resample_origin(row: usize) -> usize {
    (ROW_NUM * row) / RESAMPLE
}

/// One factor, over the single contracted axis: a stated factorization of rank one still takes the
/// separable schedule, since the per-row weight walk it caches is worth `nr` evaluations whatever
/// the rank.
#[cube]
fn resample_weights<E: Float>() -> Weights<E> {
    let mut factors = Sequence::new();
    factors.push(factor::<E>(0usize));
    Factors::new(factors)
}

#[cube(launch)]
fn resample_kernel<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    #[comptime] normalized: bool,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weights = Procedural::separable::<E, Weights<E>>(
        comptime!(space.space().subspace(&[ROW, TAP[0]])),
        resample_weights::<E>(),
    );
    let weights = if comptime!(normalized) {
        weights.normalized(comptime!(TapSupport::Whole), comptime!(DivGuard::default()))
    } else {
        weights
    }
    .tile::<E>();

    let output = output.tile(&space);
    for region in space.over(&level) {
        let mut out = output
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        out.mm(&weights.at(&region), &input.at(&region));
    }
}

#[test]
fn a_separable_lhs_contracts_a_resampling_rhs() {
    check_resampling(false);
}

#[test]
fn a_separable_resampling_lhs_normalizes_its_factor_run() {
    check_resampling(true);
}

fn check_resampling(normalized: bool) {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let in_rows = resample_origin(RROWS - 1) + RTAPS;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)]),
            Levels::leaf(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
                .walk_every(&[ROW, COL, TAP[0]])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ));

    resample_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        normalized,
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut want = 0.0f32;
            for tap in 0..RTAPS {
                let at = (resample_origin(row) + tap) * RCOLS + col;
                want += factor_value(0, tap, row) * in_data[at];
            }
            if normalized {
                want /= (0..RTAPS).map(|tap| factor_value(0, tap, row)).sum::<f32>();
            }
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "separable resample (normalized={normalized}): at ({row}, {col}) got {have}, \
                 want {want}"
            );
        }
    }
}

// ---- masked normalization against a procedural trailing tile ----------------

/// Normalize one deliberately child-local factor run at a time. This shape makes the rhs's
/// second procedural window contain one real tap and one padded tap, so `TapSupport::InBounds` must
/// distinguish the checked-read zero from an in-bounds sample without relying on backing memory.
#[cube(launch)]
fn procedural_mask_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let rhs = Procedural::new::<E, AffineCoordinate<E>>(
        comptime!(space.space().subspace(&[TAP[0], COL])),
        affine_along(TAP[0], E::new(1.0), E::new(1.0)),
    )
    .tile::<E>();
    let mut output = output.tile(&space);
    output.zero();

    for region in rhs.over(&level) {
        let rhs = rhs.at(&region);
        let child = comptime!(level.clone().child(&space.space().clone()));
        let mut factors = Sequence::new();
        factors.push(affine_along(TAP[0], E::new(1.0), E::new(0.0)));
        let weights = Procedural::separable::<E, Factors<AffineCoordinate<E>>>(
            comptime!(child.subspace(&[ROW, TAP[0]])),
            Factors::new(factors),
        )
        .normalized(
            comptime!(TapSupport::InBounds),
            comptime!(DivGuard::default()),
        )
        .tile::<E>();
        output
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD)
            .mma(&weights, &rhs);
    }
}

#[test]
fn masked_normalization_excludes_a_procedural_overhang() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![1, 1])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, 1), (COL, 1), (TAP[0], 3)]),
            Levels::leaf(&[(ROW, 1), (COL, 1), (TAP[0], 2)])
                .walk_every(&[ROW, COL, TAP[0]])
                .build(),
        ),
        Form::Static,
    );

    procedural_mask_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32).get_f32(&[0, 0]);
    // Mechanism assertion: tests that `TapSupport::InBounds` excludes the overhang tap across
    // chunks. First chunk: (1 + 2) / 2. Trailing chunk: 3 / 1 (with its padded fourth tap
    // excluded). Sum across chunks yields 1.5 + 3.0 = 4.5.
    assert!((got - 4.5).abs() < 1.0e-6, "got {got}, want 4.5");
}

// ---- masked normalization against Gmem Boundary::Zero input -----------------

#[cube(launch)]
fn resample_kernel_masked<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weights = Procedural::separable::<E, Weights<E>>(
        comptime!(space.space().subspace(&[ROW, TAP[0]])),
        resample_weights::<E>(),
    )
    .normalized(
        comptime!(TapSupport::InBounds),
        comptime!(DivGuard::default()),
    )
    .tile::<E>();

    let output = output.tile(&space);
    for region in space.over(&level) {
        let mut out = output
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        out.mm(&weights.at(&region), &input.at(&region));
    }
}

/// [`resample_kernel_masked`] with the input staged: the stage records the window it was filled
/// from, and the mask is put to that rectangle.
#[cube(launch)]
fn resample_kernel_masked_staged<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weights = Procedural::separable::<E, Weights<E>>(
        comptime!(space.space().subspace(&[ROW, TAP[0]])),
        resample_weights::<E>(),
    )
    .normalized(
        comptime!(TapSupport::InBounds),
        comptime!(DivGuard::default()),
    )
    .tile::<E>();

    let output = output.tile(&space);
    let walk = space.over(&level);
    let mut stages = Stages::smem_single(&walk, &input, StageStorage::Strided, 1usize, None);
    stages.pipelined(walk, |slot, region| {
        let mut out = output
            .at(region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        let weights = weights.at(region);
        slot.consume(|input| {
            out.mm(&weights, input);
        });
    });
}

#[test]
fn masked_normalization_dedarkens_a_boundary_zero_gmem_input() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    // Deliberately clip input rows so the last output row's tap window overhangs the edge.
    let in_rows = resample_origin(RROWS - 1) + 1;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)]),
            Levels::leaf(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
                .walk_every(&[ROW, COL, TAP[0]])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    resample_kernel_masked::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for tap in 0..RTAPS {
                let in_r = resample_origin(row) + tap;
                if in_r < in_rows {
                    let w = factor_value(0, tap, row);
                    num += w * in_data[in_r * RCOLS + col];
                    den += w;
                }
            }
            let want = if den != 0.0 { num / den } else { 0.0 };
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "masked gmem resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

/// The staged twin of [`masked_normalization_dedarkens_a_boundary_zero_gmem_input`].
///
/// `TapSupport::InBounds` has to drop the taps that overhang the input, but the fill has already
/// replaced those with zeros by the time the leaf reads them: a staged window cannot tell a padded
/// zero from a real sample, so the stage records its source window and masks to that rectangle.
///
/// The expected values are the gmem test's, because staging is a placement decision and must
/// not move a number.
#[test]
fn masked_normalization_dedarkens_a_boundary_zero_smem_input() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    // Same clipped input as the gmem twin, so the last output row's taps overhang the edge.
    let in_rows = resample_origin(RROWS - 1) + 1;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)]),
            Levels::leaf(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
                .walk_every(&[ROW, COL, TAP[0]])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    resample_kernel_masked_staged::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let mut overhanging = 0usize;
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for tap in 0..RTAPS {
                let in_r = resample_origin(row) + tap;
                if in_r < in_rows {
                    let w = factor_value(0, tap, row);
                    num += w * in_data[in_r * RCOLS + col];
                    den += w;
                } else if col == 0 {
                    overhanging += 1;
                }
            }
            let want = if den != 0.0 { num / den } else { 0.0 };
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "masked smem resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
    // Without an overhanging tap the mask is a no-op and the test would pass on a stage that
    // dropped the boundary entirely, which is the bug this defends.
    assert!(
        overhanging > 0,
        "masked smem resample: no tap overhangs the input, so the mask was never exercised"
    );
}

// ---- column-spanning normalized separable contraction -----------------------

#[cube(launch)]
fn column_spanning_resample_kernel<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weights = Procedural::separable::<E, Weights<E>>(
        comptime!(space.space().subspace(&[ROW, COL, TAP[0]])),
        resample_weights::<E>(),
    )
    .normalized(comptime!(TapSupport::Whole), comptime!(DivGuard::default()))
    .tile::<E>();

    let output = output.tile(&space);
    for region in space.over(&level) {
        let mut out = output
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        out.mm(&weights.at(&region), &input.at(&region));
    }
}

#[test]
fn a_column_spanning_separable_lhs_normalizes_its_factor_run() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    // `Weights` only reads `TAP[0]` and `ROW`, so adding `COL` to the LHS nest isolates the
    // column-spanning coordinate schedule without changing the mathematical values.
    let in_rows = resample_origin(RROWS - 1) + RTAPS;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)]),
            Levels::leaf(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
                .walk_every(&[ROW, COL, TAP[0]])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ));

    column_spanning_resample_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut want = 0.0f32;
            for tap in 0..RTAPS {
                let at = (resample_origin(row) + tap) * RCOLS + col;
                want += factor_value(0, tap, row) * in_data[at];
            }
            want /= (0..RTAPS).map(|tap| factor_value(0, tap, row)).sum::<f32>();
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "column spanning separable resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

#[cube(launch)]
fn column_spanning_resample_kernel_masked<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let weights = Procedural::separable::<E, Weights<E>>(
        comptime!(space.space().subspace(&[ROW, COL, TAP[0]])),
        resample_weights::<E>(),
    )
    .normalized(
        comptime!(TapSupport::InBounds),
        comptime!(DivGuard::default()),
    )
    .tile::<E>();

    let output = output.tile(&space);
    for region in space.over(&level) {
        let mut out = output
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        out.mm(&weights.at(&region), &input.at(&region));
    }
}

#[test]
fn a_column_spanning_separable_lhs_masks_and_dedarkens_boundary_zero_gmem_input() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    // Deliberately clip input rows so trailing output rows overhang the Boundary::Zero edge.
    let in_rows = resample_origin(RROWS - 1) + 1;
    let in_shape = shape![in_rows, RCOLS];
    let in_data = ramp(in_shape.num_elements());
    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![RROWS, RCOLS])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)]),
            Levels::leaf(&[(ROW, RROWS), (COL, RCOLS), (TAP[0], RTAPS)])
                .walk_every(&[ROW, COL, TAP[0]])
                .build(),
        ),
        Form::Static,
    );

    let in_spec = TileSpec::new(Projection::new(
        &[ROW, TAP[0], COL],
        &[
            PhysicalAxisMap::affine(&[(ROW, ROW_NUM), (TAP[0], RESAMPLE)]).over(RESAMPLE),
            PhysicalAxisMap::of(COL),
        ],
    ))
    .boundary(BoundaryPolicy::Every(Boundary::Zero));

    column_spanning_resample_kernel_masked::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(in_handle.binding().into_tensor_arg(), f32_ty, 1, in_spec),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for row in 0..RROWS {
        for col in 0..RCOLS {
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for tap in 0..RTAPS {
                let in_r = resample_origin(row) + tap;
                if in_r < in_rows {
                    let w = factor_value(0, tap, row);
                    num += w * in_data[in_r * RCOLS + col];
                    den += w;
                }
            }
            let want = if den != 0.0 { num / den } else { 0.0 };
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-4,
                "column spanning masked gmem resample: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}

// ---- zero factor sum taking fallback without poisoning siblings -------------

#[cube(launch)]
fn zero_sum_fallback_kernel<E: Float>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let mut factors = Sequence::new();
    // Factor 0: taps at k=0 (1.0) and k=1 (-1.0), sum = 0.0
    factors.push(affine_along(TAP[0], E::new(1.0), E::new(-2.0)));
    // Factor 1: taps at k=0 (2.0) and k=1 (2.0), sum = 4.0
    factors.push(affine_along(TAP[1], E::new(2.0), E::new(0.0)));
    let weights = Procedural::separable::<E, Factors<AffineCoordinate<E>>>(
        comptime!(space.space().subspace(&[ROW, TAP[0], TAP[1]])),
        Factors::new(factors),
    )
    .normalized(
        comptime!(TapSupport::Whole),
        comptime!(DivGuard {
            epsilon: 1.0e-7,
            fallback: 3.0,
        }),
    )
    .tile::<E>();

    let output = output.tile(&space);
    for region in space.over(&level) {
        let mut out = output
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        out.mm(&weights.at(&region), &input.at(&region));
    }
}

#[test]
fn a_zero_factor_sum_takes_fallback_without_poisoning_siblings() {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    // Varies along TAP[0] so factor 0's antisymmetric taps do not cancel: the result then pins
    // the fallback value itself, not just the absence of a NaN.
    let in_handle = TestInput::builder(client.clone(), shape![2, 2, 1])
        .dtype(f32_ty)
        .custom(vec![1.0f32, 1.0, 2.0, 2.0])
        .generate_without_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![1, 1])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(ROW, 1), (COL, 1), (TAP[0], 2), (TAP[1], 2)]),
            Levels::leaf(&[(ROW, 1), (COL, 1), (TAP[0], 2), (TAP[1], 2)])
                .walk_every(&[ROW, COL, TAP[0], TAP[1]])
                .build(),
        ),
        Form::Static,
    );

    zero_sum_fallback_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            in_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[TAP[0], TAP[1], COL]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32).get_f32(&[0, 0]);
    // Factor 0 has sum 0.0 -> fallback recip 3.0 -> taps become [3.0, -3.0].
    // Factor 1 has sum 4.0 -> recip 0.25 -> taps become [0.5, 0.5].
    // Product sum = (3.0 * 1.0 - 3.0 * 2.0) * (0.5 + 0.5) = -3.0.
    assert!((got + 3.0).abs() < 1.0e-6, "got {got}, want -3.0");
}

/// [`separable_kernel`] against a packed rhs scaled per tensor, decoded into a stage first:
/// `weights · stage`, the stage filled from `input ⊗ scale`. A separable contraction reads its
/// operands cell by cell (the N-D nest), which takes plain values, so the rhs decodes where the
/// kernel stages it.
#[cube(launch)]
fn separable_scaled_kernel<E: Float>(
    input: &TileArg<'_>,
    scale: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let scale = scale.tile(&space);
    let weight_axes = comptime!([&[ROW], TAP.as_slice()].concat());
    let weights = Procedural::separable::<E, Weights<E>>(
        comptime!(space.space().subspace(&weight_axes)),
        weights::<E>(),
    )
    .tile::<E>();

    let output = output.tile(&space);
    let walk = space.over(&level);
    let decoded = input.mul(&scale);
    let mut stages = Stages::smem_single(&walk, &decoded, StageStorage::Strided, 1usize, None);
    stages.pipelined(walk, |slot, region| {
        let mut out = output
            .at(region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        let weights = weights.at(region);
        slot.consume(|input| {
            out.mm(&weights, input);
        });
    });
}

/// A separable lhs against a packed, per-tensor scaled rhs, decoded into the stage the separable
/// contraction reads.
#[test]
fn a_separable_lhs_contracts_a_packed_scaled_rhs() {
    use crate::support::{TileInput, ValidationResult};
    use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};

    let client = cubecl::test_device().client();
    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S);
    let pack = scheme.num_quants();
    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below the packing factor ({pack})"
        ))
        .enforce();
        return;
    }

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[
                (ROW, ROWS),
                (COL, pack),
                (TAP[0], TAPS[0]),
                (TAP[1], TAPS[1]),
                (TAP[2], TAPS[2]),
            ]),
            Levels::leaf(&[
                (ROW, ROWS),
                (COL, pack),
                (TAP[0], TAPS[0]),
                (TAP[1], TAPS[1]),
                (TAP[2], TAPS[2]),
            ])
            .walk_every(&[ROW, COL, TAP[0], TAP[1], TAP[2]])
            .build(),
        ),
        Form::Static,
    );
    let input_axes = [TAP[0], TAP[1], TAP[2], COL];
    let input = TileInput::builder(&client, launcher.space().subspace(&input_axes))
        .untiled()
        .packed(&scheme)
        .arange();

    let f32_ty = f32::elem_type_native();
    let out_handle = TestInput::builder(client.clone(), shape![ROWS, pack])
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    // One scale for the whole rhs: a scale tile over none of its axes.
    let per_tensor = Projection::new(&input_axes, &[PhysicalAxisMap::broadcast()]);
    separable_scaled_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        input.values_arg(1, TileSpec::direct(&input_axes)),
        input.scales_arg(1, TileSpec::new(per_tensor)),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            pack,
            TileSpec::direct(&[ROW, COL]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    let scale = input.scale_values[0];
    for row in 0..ROWS {
        for col in 0..pack {
            let mut want = 0.0f32;
            for k0 in 0..TAPS[0] {
                for k1 in 0..TAPS[1] {
                    for k2 in 0..TAPS[2] {
                        let weight = factor_value(0, k0, row)
                            * factor_value(1, k1, row)
                            * factor_value(2, k2, row);
                        let at = ((k0 * TAPS[1] + k1) * TAPS[2] + k2) * pack + col;
                        want += weight * input.q[at] as f32 * scale;
                    }
                }
            }
            let have = got.get_f32(&[row, col]);
            assert!(
                (have - want).abs() < 1e-3,
                "separable scaled: at ({row}, {col}) got {have}, want {want}"
            );
        }
    }
}
