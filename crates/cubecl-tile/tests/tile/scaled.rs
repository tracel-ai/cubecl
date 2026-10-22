//! `c.mm(&a.mul(&s), &b)`: the contraction with one
//! factor scaled by a **real operand**, on the factor the kernel wrote it on.
//!
//! *Which* operand is not stated: the scales' own axes say it. A scale over the output's columns
//! is a fact about the rhs's columns and nothing else could fold it in; anything else scales the
//! lhs. One verb serves both: `(a ⊗ s) · b` and `a · (b ⊗ s)` are one sum, folded where cheapest.
//!
//! The point is what the kernel signature says: the values are a tensor of values, the scales a
//! tensor of scales, both named at the call, and the fold is a verb the kernel writes. Nothing
//! decodes behind a read or rides a side channel, so `f16` scales work with no widening pass.
//!
//! The scales resolve at their own granularity through their own projection: a plain `KB` for a
//! per-block scale, `KI` too for a per-element one, an omitted axis for a broadcast.

use crate::support::{HostData, HostDataType, TestInput, skip_unless_plane_holds};
use cubecl::{
    bytes::Bytes,
    prelude::*,
    quant::scheme::{QuantValue, ScaleDtype},
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubecl_common::{e2m1, e4m3};
use cubecl_tile::Instruction;
use cubecl_tile::kind::Field;
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::layout::split;
use cubecl_tile::stage::{Prefetch, RowChunks, UnitRead};
use cubecl_tile::*;
use half::f16;

use super::matmul::{require_cmma_8x8x8, require_cmma_8x8x8_f32};
use super::{Form, implied};
use cubecl_tile::launch::Bound;

/// Which factor a test kernel writes its scales on. The engine has no such enum: a kernel says
/// which by where it writes `.mul()`, and these kernels serve both cases from one launch.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Scaled {
    Lhs,
    Rhs,
}

const M: Axis = Axis(0);
const N: Axis = Axis(1);
/// The contraction, as the two axes a block makes of it: which block, and where inside it.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// The register block the contractions here run under, but for the wide-lhs one.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `c = (a ⊗ s) · b`, with `s` at whatever granularity its own projection states.
#[cube(launch)]
fn scaled_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        match comptime!(side) {
            Scaled::Lhs => c_r.mma(&a.at(&region).mul(&scale.at(&region)), &b.at(&region)),
            Scaled::Rhs => c_r.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region))),
        }
    }
}

/// [`scaled_matmul`] with the accumulator promoted to registers: the partials never round-trip
/// through `c`'s element between `K` steps, which is the form a decode gemv wants.
#[cube(launch)]
fn scaled_matmul_promoted<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Registers {
            config: REGISTER_BLOCK
        }),
        Semiring::SUM_PROD,
    );
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        match comptime!(side) {
            Scaled::Lhs => acc_r.mma(&a.at(&region).mul(&scale.at(&region)), &b.at(&region)),
            Scaled::Rhs => acc_r.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region))),
        }
    }
    acc.drained_into(&c);
}

/// `c = (a ⊗ blocks ⊗ global) · b`, written as a kernel writes any sum: what `factor` takes out of
/// the operand for a sum in `instruction`'s form multiplies the sum once, as it drains; what it
/// leaves rides the load.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn factored_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    blocks: &TileArg<'_>,
    global: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] instruction: Instruction,
    #[define(E)] _dtype: ElemType,
) {
    let a = a
        .tile(&space)
        .mul(&blocks.tile(&space))
        .mul(&global.tile(&space));
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    let (a, a_factor) = a.factor(c.space(), instruction);
    let acc = c.accumulator::<E>(&a, &b, instruction, Semiring::SUM_PROD);
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        acc_r.mma(&a.at(&region), &b.at(&region));
    }
    c.dequantize_from(&acc.scaled(&a_factor));
}

/// [`scaled_matmul`] with two scale levels: block scales, and one factor over the whole tensor.
#[cube(launch)]
fn two_level_scaled_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    blocks: &TileArg<'_>,
    global: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    // The operand is the hierarchy: block scales, under the factor that covers a tile of their
    // tiles. Nothing states a scheme.
    let blocks = blocks.tile(&space);
    let global = global.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        match comptime!(side) {
            Scaled::Lhs => c_r.mma(
                &a.at(&region)
                    .mul(&blocks.at(&region))
                    .mul(&global.at(&region)),
                &b.at(&region),
            ),
            Scaled::Rhs => c_r.mma(
                &a.at(&region),
                &b.at(&region)
                    .mul(&blocks.at(&region))
                    .mul(&global.at(&region)),
            ),
        }
    }
}

/// [`scaled_matmul`] on a tensor-core accumulator: the scaled operand is landed in shared memory
/// by the plane's units, unpacked and scaled, and loaded as the fragment the plain instruction
/// takes. Both operands carry a landing here so one kernel serves either side.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn scaled_matmul_cmma<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    let acc = c.accumulator::<E>(&a, &b, Instruction::Cmma, Semiring::SUM_PROD);
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        match comptime!(side) {
            Scaled::Lhs => acc_r.mma(&a.at(&region).mul(&scale.at(&region)), &b.at(&region)),
            Scaled::Rhs => acc_r.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region))),
        }
    }
    acc.drained_into(&c);
}

/// **Two scale levels, applied in order.** `nvfp4`'s shape: a scale per block of the contraction,
/// under one factor for the whole tensor.
///
/// The second level is an operand like the first, spanning the same axes and distinguishing
/// fewer of them, and it rides the same [`Scales`] as the block level. Nothing in the kernel
/// counts levels: it windows `scales` and hands it to the leaf.
#[test]
fn two_levels_fold_in_order() {
    let (rows, cols, block, blocks) = (4, 4, 8, 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let g = vec![0.25f32];

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let (g_t, _) = TestInput::builder(client.clone(), shape![1])
        .dtype(dtype)
        .custom(g.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    two_level_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            g_t.binding().into_tensor_arg(),
            dtype,
            1,
            // Same axes as the level below it, addressing neither: one value over all their tiles.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::broadcast()])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * g[0] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`factored_matmul`] in registers: the operand is decoded as it is loaded under its block scales,
/// and its global, which `factor` takes out for a sum scaled in its registers, scales the sum; the
/// result is the same product.
#[test]
fn a_decoded_operands_global_scale_leaves_for_the_sum() {
    check_factored(
        Instruction::Registers {
            config: REGISTER_BLOCK,
        },
        4,
    );
}

/// [`factored_matmul`] in cmma fragments: a cmma sum is scaled only through shared memory, so the
/// operand keeps its global and lands decoded under every scale, and the fragments drain unscaled.
#[test]
fn a_decoded_operand_keeps_its_global_scale_on_the_load_into_cmma_fragments() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    check_factored(Instruction::Cmma, 8);
}

/// `c = (a ⊗ blocks ⊗ global) · b` over `size × size` summed in `instruction`'s form, written
/// through `factor` and `scaled` as a kernel writes any sum, against the host product.
fn check_factored(instruction: Instruction, size: usize) {
    let (rows, cols, block, blocks) = (size, size, 8, 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let g = vec![0.25f32];

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let (g_t, _) = TestInput::builder(client.clone(), shape![1])
        .dtype(dtype)
        .custom(g.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    let a_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    factored_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(a_projection.clone()),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(a_projection.scales_per(KB)),
        ),
        TileArgLaunch::new(
            g_t.binding().into_tensor_arg(),
            dtype,
            1,
            // Same axes as the level below it, addressing neither: one value over all their tiles.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::broadcast()])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        instruction,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * g[0] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// One block per region, so each region carries one scale per row.
#[test]
fn a_scaled_contraction_folds_the_block_scale_in() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    // Small integers, so the reference is exact.
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    // The values' projection names the block; the scales' is derived from it, one per `KB`.
    let a_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(a_projection.clone()),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(a_projection.scales_per(KB)),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// A region inside one block: several of them share a scale, because they share a `KB`.
#[test]
fn a_cut_finer_than_the_block_reuses_its_scale() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block / 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **A scale over no axis at all.** The base of the hierarchy: one number covering everything,
/// which is what a per-tensor level is.
///
/// Spelled the way every other granularity is, by which axes the operand distinguishes. The nest
/// still names `[M, KB]`, so the scales' matrix has the shape any other level's would; the
/// projection addresses neither, so every position reads the same value. Nothing says "per tensor".
#[test]
fn a_scale_over_no_axis_covers_everything() {
    let (rows, cols, block, blocks) = (4, 4, 8, 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s = vec![0.5f32];

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![1])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            // One physical axis, addressing no logical one: every position resolves to index 0.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::broadcast()])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[0] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// A region spanning two blocks: the scale changes within it, and the step's own `KB` coordinate
/// is what picks which.
#[test]
fn a_cut_coarser_than_the_block_changes_scale_within_a_region() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (2, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The one that pays for the design.** The scales are an `f16` tensor and the kernel reads them
/// as one: no widening pass, no scheme saying otherwise, no way for the two to disagree. A scale
/// is whatever its tensor holds because it is a tensor. The values stay `f32`, `s` exact in both.
#[test]
fn f16_scales_are_read_as_f16() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let scale_dtype = f16::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(scale_dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            scale_dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The scales span `N`, so they scale the rhs, and the call site says so
/// on the rhs; the leaf checks that against the axes.
#[test]
fn scales_over_the_columns_scale_the_rhs() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            // Spanning `N` is what makes it the rhs's.
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Rhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The rhs scale under a cut finer than its block: several regions share it.
#[test]
fn an_rhs_scale_survives_a_finer_cut() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block / 2);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Rhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// And under a coarser one, where the scale changes within a region.
#[test]
fn an_rhs_scale_changes_within_a_coarser_region() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (2, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Rhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The gap the decode gemv was waiting on.** The kernel opens a register block above the whole
/// walk, so the scaled steps fold into it directly: refused before, and the same numbers as the
/// memory-backed form.
#[test]
fn a_promoted_accumulator_takes_the_scaled_contraction() {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    scaled_matmul_promoted::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB],
                &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`scaled_matmul_promoted`] with the rhs's scales served as lines: several of them per read,
/// along `N`.
#[cube(launch)]
fn wide_rhs_scaled_matmul_promoted<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Registers {
            config: REGISTER_BLOCK
        }),
        Semiring::SUM_PROD,
    );
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        match comptime!(side) {
            Scaled::Lhs => acc_r.mma(&a.at(&region).mul(&scale.at(&region)), &b.at(&region)),
            Scaled::Rhs => acc_r.mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region))),
        }
    }
    acc.drained_into(&c);
}

/// **Scales served as lines along the columns, into a promoted accumulator.** The twin of
/// [`lhs_scales_are_served_several_at_a_time`].
///
/// The case the two old asserts disagreed about: the memory nest admitted a wide scale line where
/// its step folded one contracted value, the promoted block where the scales rode the rhs, and
/// nothing exercised the second.
///
/// The block walks its columns under a constant ordinal, which is what a wide read needs — a fold
/// is a unit of the read it arrived in, and a unit index is not addressable at runtime. So unit
/// `j` of a scale line goes with column `j`, and the scales vary per `(block of K, column)`.
#[test]
fn rhs_scales_are_served_several_at_a_time() {
    let (rows, cols, block, blocks, plane_units) = (2, 4, 8, 4, 4);
    let (per_region, inside) = (1, block);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Distinct per (block of `K`, column), halves so the reference is exact.
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, per_region), (KI, inside)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    wide_rhs_scaled_matmul_promoted::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            plane_units,
            // `N` innermost: the width lands on the axis the scales vary over fastest, and
            // spanning it is what makes them the rhs's.
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Rhs,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| a[m * depth + k] * b[k * cols + n] * s[(k / block) * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// [`scaled_matmul`] with the lhs's scales served as lines: several of them per read, along `KB`.
/// The fold rides the unit walk, which reads one line per `lw` steps and takes a fixed component
/// of it; the scalar walk has no line ordinal to fold under, so this block fans out over units.
#[cube(launch)]
fn wide_lhs_scaled_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] side: Scaled,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region).accumulating(
            comptime!(RegisterBlock::new(64).component_fanout()),
            Semiring::SUM_PROD,
        );
        match comptime!(side) {
            Scaled::Lhs => c_r.mma(&a.at(&region).mul(&scale.at(&region)), &b.at(&region)),
            Scaled::Rhs => c_r
                .accumulating(
                    comptime!(RegisterBlock::new(64).component_fanout()),
                    Semiring::SUM_PROD,
                )
                .mma(&a.at(&region), &b.at(&region).mul(&scale.at(&region))),
        }
    }
}

/// **Scales served as lines along the contraction.** The lhs's scales vary over `KB` and nothing
/// else, so `KB` is their innermost axis and a read serves several blocks of `K` at once.
///
/// The block walks its contraction in runs of one scale line for this: the folds are unrolled so
/// each one's unit is a constant, and the lines under one fold stay rolled, since they all take
/// the same scale. One row here, so the scales are per block of `K` alone.
#[test]
fn lhs_scales_are_served_several_at_a_time() {
    let (cols, block, blocks, plane_units) = (4, 4, 4, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    // Distinct per block of `K`, halves so the reference is exact.
    let s: Vec<f32> = (0..blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![1, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![1, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, 1), (N, cols), (KB, blocks), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );

    wide_lhs_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            4,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            plane_units,
            // `KB` innermost and alone: the width lands on the axis the scales vary over.
            TileSpec::new(Projection::new(&[M, KB], &[PhysicalAxisMap::of(KB)])),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Scaled::Lhs,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for n in 0..cols {
        let want: f32 = (0..depth)
            .map(|k| a[k] * s[k / block] * b[k * cols + n])
            .sum();
        let have = got.get_f32(&[0, n]);
        assert!(
            (have - want).abs() < 1e-3,
            "at {n}: got {have}, want {want}"
        );
    }
}

/// Which operand of the tensor-core test carries the scales, and how the rhs is stored.
#[derive(Clone, Copy)]
enum CmmaCase {
    /// Scales per `(row, block of K)`, on the lhs.
    Lhs,
    /// Scales per `(block of K, column)`, on a rhs stored `{K, N}`.
    RhsRowMajor,
    /// The same scales on a rhs stored `{N, K}`, read col-major, so the landing is the
    /// transpose of the scales' matrix.
    RhsColMajor,
}

/// `c = (a ⊗ s) · b` or `a · (b ⊗ s)` through the M2's `8x8x8` fragment, one block of `K` a
/// region, against the host product.
fn check_scaled_cmma(case: CmmaCase) {
    let (rows, cols, block, blocks) = (8, 8, 8, 4);
    let depth = block * blocks;

    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..blocks * 8).map(|i| (i as f32 + 1.0) / 2.0).collect();
    // The rhs as stored: `{K, N}`, or `{N, K}` holding the same matrix.
    let b_stored: Vec<f32> = match case {
        CmmaCase::RhsColMajor => (0..cols * depth)
            .map(|i| b[(i % depth) * cols + i / depth])
            .collect(),
        _ => b.clone(),
    };
    let b_shape = match case {
        CmmaCase::RhsColMajor => shape![cols, depth],
        _ => shape![depth, cols],
    };
    let s_shape = match case {
        CmmaCase::Lhs => shape![rows, blocks],
        _ => shape![blocks, cols],
    };

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), b_shape)
        .dtype(dtype)
        .custom(b_stored)
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), s_shape)
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );
    let split = PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]);
    let b_spec = match case {
        CmmaCase::RhsColMajor => TileSpec::new(Projection::new(
            &[N, KB, KI],
            &[PhysicalAxisMap::of(N), split.clone()],
        )),
        _ => TileSpec::new(Projection::new(
            &[KB, KI, N],
            &[split.clone(), PhysicalAxisMap::of(N)],
        )),
    };
    let s_spec = match case {
        CmmaCase::Lhs => TileSpec::new(Projection::new(
            &[M, KB],
            &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(KB)],
        )),
        _ => TileSpec::new(Projection::new(
            &[KB, KI, N],
            &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
        )),
    };
    let side = match case {
        CmmaCase::Lhs => Scaled::Lhs,
        _ => Scaled::Rhs,
    };

    scaled_matmul_cmma::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), split],
            )),
        ),
        TileArgLaunch::new(b_t.binding().into_tensor_arg(), dtype, 1, b_spec),
        TileArgLaunch::new(s_t.binding().into_tensor_arg(), dtype, 1, s_spec),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        side,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    let scale = match case {
                        CmmaCase::Lhs => s[m * blocks + k / block],
                        _ => s[(k / block) * cols + n],
                    };
                    a[m * depth + k] * scale * b[k * cols + n]
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// **The scaled contraction runs on the tensor cores.** The lhs is landed scaled by the plane's
/// units and loaded as the `A` fragment; the instruction is the plain one.
#[test]
fn a_cmma_accumulator_takes_the_scaled_contraction() {
    check_scaled_cmma(CmmaCase::Lhs);
}

/// The rhs side of the same, stored `{K, N}`: the landing is the `B` fragment as stored.
#[test]
fn a_cmma_accumulator_takes_rhs_scales() {
    check_scaled_cmma(CmmaCase::RhsRowMajor);
}

/// The rhs stored `{N, K}`, the way a weight lies with its contraction innermost: the landing is
/// col-major, so each line's scale sits at its column and its row's block.
#[test]
fn a_cmma_accumulator_takes_rhs_scales_col_major() {
    check_scaled_cmma(CmmaCase::RhsColMajor);
}

/// [`scaled_matmul_cmma`] with the rhs **staged as its words** before it lands: the packed
/// window is copied into shared memory verbatim, the stage keeps the packing, and the landing
/// unpacks and scales out of the stage exactly as it would out of the global window.
#[cube(launch)]
fn scaled_matmul_cmma_staged<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    let mut stage = b.stage(comptime!(level.clone()), StageStorage::Strided);
    let acc = c.accumulator::<E>(&a, &b, Instruction::Cmma, Semiring::SUM_PROD);
    for region in space.over(&level) {
        stage.copy_from(&b.at(&region));
        sync_cube();
        let mut acc_r = acc.at(&region);
        acc_r.mma(&a.at(&region), &stage.mul(&scale.at(&region)));
        // The stage is refilled next region, once every plane has landed from it.
        sync_cube();
    }
    acc.drained_into(&c);
}

/// **A packed stage lands on the tensor cores.** `e2m1` values eight to a word, stored `{N, K}`
/// with the contraction innermost as a weight lies, staged one scale block at a time as the words
/// they are, then unpacked and scaled into the fragment's landing: a quarter the stage, one answer.
#[test]
fn a_packed_stage_lands_on_the_tensor_cores() {
    let (rows, cols, block, blocks) = (8, 8, 8, 4);
    let depth = block * blocks;
    let field = QuantValue::E2M1;
    let factor = 32 / field.size_bits();

    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    // The rhs as stored, `{N, K}`: every `e2m1` code, cycled, eight to a word down `k`.
    let codes: Vec<u32> = (0..cols * depth).map(|i| (i % 16) as u32).collect();
    let words: Vec<u32> = codes
        .chunks(factor)
        .map(|word| {
            word.iter()
                .enumerate()
                .fold(0u32, |acc, (j, &c)| acc | (c << (j * field.size_bits())))
        })
        .collect();
    let s: Vec<f32> = (0..blocks * cols).map(|i| (i as f32 + 1.0) / 2.0).collect();

    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let b_t = TensorHandle::new_contiguous(
        vec![cols, depth / factor],
        client.create(Bytes::from_elems(words)),
        u32::elem_type_native(),
    );
    let (s_t, _) = TestInput::builder(client.clone(), shape![blocks, cols])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );
    let split = PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]);

    scaled_matmul_cmma_staged::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[PhysicalAxisMap::of(M), split.clone()],
            )),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            u32::elem_type_native(),
            1,
            TileSpec::new(Projection::new(
                &[N, KB, KI],
                &[PhysicalAxisMap::of(N), split],
            ))
            .packed(field),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[PhysicalAxisMap::of(KB), PhysicalAxisMap::of(N)],
            )),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth)
                .map(|k| {
                    let b = e2m1::from_bits(codes[n * depth + k] as u8).to_f32();
                    a[m * depth + k] * s[(k / block) * cols + n] * b
                })
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The columns, as the two axes a storage tile makes of them: which tile, and where inside it.
const NB: Axis = Axis(4);
const NI: Axis = Axis(5);

/// `c = a · (b ⊗ s)` over a weight **stored in tile order**, walked the way the memory-bound
/// kernel walks it on either arm: the cube grid, the planes, the chunks a plane walks under one
/// load of its scales, the steps of a chunk, and below them the block's units or the one fragment.
///
/// The scales of a chunk are loaded once, into the plane's units or its own shared window, and
/// every step reads its scale from there.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn chunked_scaled_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] instruction: Instruction,
    #[comptime] chunks: Level,
    #[comptime] read: UnitRead,
    #[define(E)] _dtype: ElemType,
) {
    // Both factors land where the instruction reads a window as it lies, and neither does where
    // it reads through a layout: the contraction decides.
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    for cube in space {
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        let scale_cube = scale.at(&cube);
        let c_cube = c.at(&cube);
        for plane in cube {
            let a_plane = a_cube.at(&plane);
            let b_plane = b_cube.at(&plane);
            let scale_plane = scale_cube.at(&plane);
            let c_plane = c_cube.at(&plane);
            let mut lines = scale_plane.stage(
                comptime!(chunks.clone()),
                comptime!(StageStorage::Lines { read }),
            );
            let sum = c_plane.accumulator::<E>(&a_plane, &b_plane, instruction, Semiring::SUM_PROD);
            for chunk in plane {
                lines.copy_from(&scale_plane.at(&chunk));
                for step in chunk {
                    for leaf in step {
                        let mut sum_leaf = sum.at(&leaf);
                        sum_leaf.mma(&a_plane.at(&leaf), &b_plane.at(&leaf).mul(&lines.at(&leaf)));
                    }
                }
            }
            sum.drained_into(&c_plane);
        }
    }
}

/// Row-major strides over `shape`.
fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// The scales of a tile-ordered weight as a test stores them: which float, and the words.
#[derive(Clone, Copy, Debug)]
enum TileScales {
    /// Whole `f32`s, one a word.
    F32,
    /// `ue4m3` bytes, four to a word: what NVFP4 carries.
    Ue4m3,
}

/// A weight stored in tile order, with its activation and the host's view of both.
///
/// The values lie `[NB][KB][NI][KI]`: a tile is sixteen columns of sixteen `k`, a column eight
/// bytes. The scales lie `[NB][KB][NI]`: a tile's sixteen scales are one line, one per column,
/// and the tiles down `k` follow one another, so a chunk of them is consecutive lines.
struct TileOrdered {
    rows: usize,
    tile: usize,
    n_tiles: usize,
    k_tiles: usize,
    a: Vec<f32>,
    words: Vec<u32>,
    s: Vec<f32>,
}

impl TileOrdered {
    const FIELD: QuantValue = QuantValue::E2M1;

    fn new(rows: usize, n_tiles: usize, k_tiles: usize) -> Self {
        let tile = 16;
        let depth = k_tiles * tile;
        let factor = 32 / Self::FIELD.size_bits();
        let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
        let this = TileOrdered {
            rows,
            tile,
            n_tiles,
            k_tiles,
            a,
            words: Vec::new(),
            s: Vec::new(),
        };
        // The words as stored: tile order, a column's sixteen codes eight to a word, low nibble
        // first.
        let mut words = Vec::with_capacity(n_tiles * k_tiles * tile * tile / factor);
        for nb in 0..n_tiles {
            for kb in 0..k_tiles {
                for ni in 0..tile {
                    for word in 0..tile / factor {
                        words.push((0..factor).fold(0u32, |acc, j| {
                            acc | ((this.code(nb, kb, ni, word * factor + j) as u32)
                                << (j * Self::FIELD.size_bits()))
                        }));
                    }
                }
            }
        }
        // One scale per column per tile: halves from 0.5 to 4, exact in `e4m3`, and different
        // between neighbouring columns *and* neighbouring tiles, so a scale read at the wrong
        // column or the wrong block is a wrong number rather than the same one.
        let s: Vec<f32> = (0..n_tiles * k_tiles * tile)
            .map(|i| ((i % 8) + (i / 8) + (i / 64)) as f32 % 8.0 / 2.0 + 0.5)
            .collect();
        TileOrdered { words, s, ..this }
    }

    fn depth(&self) -> usize {
        self.k_tiles * self.tile
    }

    fn cols(&self) -> usize {
        self.n_tiles * self.tile
    }

    /// The `e2m1` code at `(nb, kb, ni, ki)`: every code, and a different run of them in every
    /// column of every tile, so a value read from the wrong column or the wrong tile is a wrong
    /// number rather than the same one.
    fn code(&self, nb: usize, kb: usize, ni: usize, ki: usize) -> usize {
        (ki + 3 * ni + 5 * kb + 7 * nb) % 16
    }

    /// The scale at `(nb, kb, ni)`, as stored.
    fn scale_at(&self, nb: usize, kb: usize, ni: usize) -> f32 {
        self.s[(nb * self.k_tiles + kb) * self.tile + ni]
    }

    /// The product on the host.
    fn want(&self, m: usize, n: usize) -> f32 {
        let (nb, ni) = (n / self.tile, n % self.tile);
        (0..self.depth())
            .map(|k| {
                let (kb, ki) = (k / self.tile, k % self.tile);
                let b = e2m1::from_bits(self.code(nb, kb, ni, ki) as u8).to_f32();
                self.a[m * self.depth() + k] * b * self.scale_at(nb, kb, ni)
            })
            .sum()
    }

    fn space(&self) -> Space {
        Space::new(&[
            (M, self.rows),
            (NB, self.n_tiles),
            (NI, self.tile),
            (KB, self.k_tiles),
            (KI, self.tile),
        ])
    }

    /// The activation, served at the width a word of the weight unpacks to.
    fn a_op(&self, client: &Client, launcher: &Launcher) -> Bound {
        let (a_t, _) = TestInput::builder(client.clone(), shape![self.rows, self.depth()])
            .dtype(f32::elem_type_native())
            .custom(self.a.clone())
            .generate_with_f32_host_data();
        launcher
            .arg(a_t.binding(), f32::elem_type_native())
            .gathered(
                Projection::dims()
                    .dim(M)
                    .dim(split(&[(KB, self.k_tiles), (KI, self.tile)]))
                    .build(),
            )
            .vectorize(32 / Self::FIELD.size_bits())
            .build()
            .unwrap()
    }

    /// The activation in `elem`, stated as its rows by its blocks by the position inside one, a
    /// dim an axis: the same values as [`a_op`](Self::a_op), bound directly, which a stage can
    /// hold.
    fn a_op_blocked(&self, client: &Client, launcher: &Launcher, elem: ElemType) -> Bound {
        let (a_t, _) =
            TestInput::builder(client.clone(), shape![self.rows, self.k_tiles, self.tile])
                .dtype(elem)
                .custom(self.a.clone())
                .generate_with_f32_host_data();
        launcher
            .arg(a_t.binding(), elem)
            .axes(&[M, KB, KI])
            .vectorize(32 / Self::FIELD.size_bits())
            .build()
            .unwrap()
    }

    /// The weight as stored, read `read` values a load where stated, a word otherwise. A packed
    /// binding counts values, its words being the packing's business: the shape and the strides
    /// are the tiles' in values.
    fn b_op(&self, client: &Client, launcher: &Launcher, read: Option<usize>) -> Bound {
        let shape = vec![self.n_tiles, self.k_tiles, self.tile, self.tile];
        let b_t = TensorHandle::new_contiguous(
            shape.clone(),
            client.create(Bytes::from_elems(self.words.clone())),
            u32::elem_type_native(),
        );
        let mut binding = b_t.binding();
        binding.shape = shape.clone().into();
        binding.strides = contiguous_strides(&shape).into();
        let b = launcher
            .arg(binding, u32::elem_type_native())
            .gathered(Projection::dims().dim(NB).dim(KB).dim(NI).dim(KI).build())
            .packed(Self::FIELD);
        match read {
            Some(values) => b.vectorize(values),
            None => b,
        }
        .build()
        .unwrap()
    }

    /// The scales as stored, served `served` values a read, the columns one plane holds: whole
    /// words, or `ue4m3` bytes four to a word, the same shape in values either way.
    fn s_op(
        &self,
        client: &Client,
        launcher: &Launcher,
        scales: TileScales,
        served: usize,
    ) -> Bound {
        let axes = Projection::dims()
            .dim(NB)
            .dim(KB)
            .dim(NI)
            .spanning(KI)
            .build();
        let shape = vec![self.n_tiles, self.k_tiles, self.tile];
        match scales {
            TileScales::F32 => {
                let (s_t, _) = TestInput::builder(
                    client.clone(),
                    shape![self.n_tiles, self.k_tiles, self.tile],
                )
                .dtype(f32::elem_type_native())
                .custom(self.s.clone())
                .generate_with_f32_host_data();
                launcher
                    .arg(s_t.binding(), f32::elem_type_native())
                    .gathered(axes)
                    .vectorize(served)
                    .build()
                    .unwrap()
            }
            TileScales::Ue4m3 => {
                let per_word = 4;
                let bytes: Vec<u32> = self
                    .s
                    .chunks(per_word)
                    .map(|word| {
                        word.iter().enumerate().fold(0u32, |acc, (j, &v)| {
                            acc | ((e4m3::from_f32(v).to_bits() as u32) << (j * 8))
                        })
                    })
                    .collect();
                let s_t = TensorHandle::new_contiguous(
                    shape.clone(),
                    client.create(Bytes::from_elems(bytes)),
                    u32::elem_type_native(),
                );
                let mut binding = s_t.binding();
                binding.shape = shape.clone().into();
                binding.strides = contiguous_strides(&shape).into();
                launcher
                    .arg(binding, u32::elem_type_native())
                    .gathered(axes)
                    .packed(Field::of_scale(ScaleDtype::UE4M3))
                    .vectorize(served)
                    .build()
                    .unwrap()
            }
        }
    }

    fn c_op(&self, launcher: &Launcher, c: &TensorHandle) -> Bound {
        launcher
            .arg(c.clone().binding(), c.dtype)
            .gathered(
                Projection::dims()
                    .dim(M)
                    .dim(split(&[(NB, self.n_tiles), (NI, self.tile)]))
                    .build(),
            )
            .build()
            .unwrap()
    }

    fn check(&self, client: &Client, c: TensorHandle, what: &str) {
        let got = HostData::from_tensor_handle(client, c, HostDataType::F32);
        for m in 0..self.rows {
            for n in 0..self.cols() {
                let want = self.want(m, n);
                let have = got.get_f32(&[m, n]);
                assert!(
                    (have - want).abs() < 1e-2 * want.abs().max(1.0),
                    "{what} at ({m}, {n}): got {have}, want {want}"
                );
            }
        }
    }
}

/// Which arm a chunked contraction runs on.
#[derive(Clone, Copy, Debug)]
enum Arm {
    Registers,
    Landing,
}

/// **A plane holds its scales for a chunk.** The weight lies in tile order and its scales in
/// lines, `[NB][KB][NI]`; a plane walks the contraction a chunk of thirty-two blocks at a time and
/// loads the chunk's thirty-two lines once: unit `t` holds line `t`, or a shared window holds all.
///
/// Every step reads the scale of the value it lands or contracts at that value's coordinates, a
/// word at a time.
///
/// On the register arm a unit holds one column over one block a step, the plane's units are a
/// tile's columns by two blocks, and a chunk is sixteen steps. On the tensor cores a plane holds
/// one fragment, eight rows by half a tile's columns, and walks a chunk one fragment depth a step.
fn check_chunked(arm: Arm, scales: TileScales, read: UnitRead) {
    let (rows, n_tiles, chunk, chunks) = (8, 2, 32, 2);
    let w = TileOrdered::new(rows, n_tiles, chunk * chunks);
    let client = cubecl::test_device().client();
    if matches!(arm, Arm::Landing) && !require_cmma_8x8x8_f32(&client) {
        return;
    }
    // The register arm distributes a tile's columns by two blocks to the plane's units.
    if matches!(arm, Arm::Registers) && skip_unless_plane_holds(&client, (w.tile * 2) as u32) {
        return;
    }
    let dtype = f32::elem_type_native();
    let c = TestInput::builder(client.clone(), shape![rows, w.cols()])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // Leaf up: what a unit or a plane holds a step, the steps of a chunk, the chunks, the planes
    // of a cube, the cubes. A fragment is a level of one all the same, since the body's innermost
    // loop is a level's.
    let levels = match arm {
        Arm::Registers => Levels::leaf(&[(M, rows), (NI, 1), (KB, 1), (KI, w.tile)])
            .units(&[(NI, w.tile), (KB, 2)])
            .walk(&[(KB, chunk / 2)])
            .walk_every(&[KB])
            .planes(&[(NB, 1)])
            .cubes(&[NB])
            .build(),
        Arm::Landing => Levels::leaf(&[(M, rows), (NI, 8), (KI, 8)])
            .walk(&[(NI, 1)])
            .walk(&[(KB, chunk), (KI, 2)])
            .walk_every(&[KB])
            .planes(&[(NI, w.tile / 8)])
            .cubes(&[NB])
            .build(),
    };
    let chunks_level = levels[2].clone();
    let instruction = match arm {
        Arm::Registers => Instruction::Registers {
            config: REGISTER_BLOCK,
        },
        Arm::Landing => Instruction::Cmma,
    };
    let launcher = implied(&client, Partitioning::new(w.space(), levels), Form::Static);
    let s_op = w.s_op(
        &client,
        &launcher,
        scales,
        match arm {
            Arm::Registers => w.tile,
            // The landing steps a plane's eight columns of a tile, a line of scales at a time.
            Arm::Landing => 8,
        },
    );

    chunked_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w.a_op(&client, &launcher).arg(),
        w.b_op(&client, &launcher, None).arg(),
        s_op.arg(),
        w.c_op(&launcher, &c).arg(),
        launcher.partitioning_arg(),
        instruction,
        chunks_level,
        read,
        dtype,
    );
    w.check(&client, c, &format!("{arm:?} {scales:?}"));
}

/// Every arm and every scale element, **read both ways**: what the units hold is the same
/// either way, so a value read by shuffle and a value read out of the plane's window are the
/// same value or one of the two is wrong.
#[test]
fn a_plane_holds_its_scales_in_its_units() {
    for read in [UnitRead::Shuffle, UnitRead::PlaneShared] {
        check_chunked(Arm::Registers, TileScales::F32, read);
    }
}

#[test]
fn a_plane_holds_its_byte_scales_in_its_units() {
    for read in [UnitRead::Shuffle, UnitRead::PlaneShared] {
        check_chunked(Arm::Registers, TileScales::Ue4m3, read);
    }
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores() {
    for read in [UnitRead::Shuffle, UnitRead::PlaneShared] {
        check_chunked(Arm::Landing, TileScales::F32, read);
    }
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores_under_byte_scales() {
    for read in [UnitRead::Shuffle, UnitRead::PlaneShared] {
        check_chunked(Arm::Landing, TileScales::Ue4m3, read);
    }
}

/// `c = a · (b ⊗ s)` over a weight stored in tile order, walked as the compute-bound body walks
/// it on the tensor cores: a plane holds a partition of fragments (two rows of two columns), the
/// weight unpacked and scaled by the chunk's lines.
///
/// The partition's fragments load through their landing a depth at a time, the quant block being
/// the loop inside the partition's depth: two instructions under one scale.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn partitioned_scaled_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] chunks: Level,
    #[comptime] read: UnitRead,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    // The weight is scaled at every step, which lands it on its own.
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    for cube in space {
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        let scale_cube = scale.at(&cube);
        let c_cube = c.at(&cube);
        for plane in cube {
            let a_plane = a_cube.at(&plane);
            let b_plane = b_cube.at(&plane);
            let scale_plane = scale_cube.at(&plane);
            let c_plane = c_cube.at(&plane);
            let mut lines = scale_plane.stage(
                comptime!(chunks.clone()),
                comptime!(StageStorage::Lines { read }),
            );
            let sum =
                c_plane.accumulator::<E>(&a_plane, &b_plane, Instruction::Cmma, Semiring::SUM_PROD);
            for chunk in plane {
                lines.copy_from(&scale_plane.at(&chunk));
                for step in chunk {
                    let a_step = a_plane.at(&step);
                    let b_step = b_plane.at(&step).mul(&lines.at(&step));
                    for block in step.walk().unrolled() {
                        for depth in block.walk().unrolled() {
                            let mut sum_depth = sum.at(&depth);
                            sum_depth.mma(&a_step.at(&depth), &b_step.at(&depth));
                        }
                    }
                }
            }
            sum.drained_into(&c_plane);
        }
    }
}

/// **A tile-ordered weight lands on the tensor cores under a partition.** Sixteen rows: a plane
/// holds two fragments of rows by two of columns and walks the contraction two blocks a step,
/// two instructions a block, landing every step's window once.
fn check_partitioned(scales: TileScales, read: UnitRead) {
    let (rows, n_tiles, chunk, chunks, fragment) = (16, 2, 32, 2, 8);
    let w = TileOrdered::new(rows, n_tiles, chunk * chunks);
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let c = TestInput::builder(client.clone(), shape![rows, w.cols()])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // Leaf up: the instruction; the grid of fragments a plane holds at one depth; the quant
    // block, two instructions deep under one scale; the partition's depth, two blocks a step;
    // the steps of a chunk; the chunks; the planes; the cubes.
    let levels = Levels::leaf(&[(M, fragment), (NI, fragment), (KI, fragment)])
        .walk(&[(M, rows / fragment), (NI, w.tile / fragment)])
        .walk(&[(KI, w.tile / fragment)])
        .walk(&[(KB, 2)])
        .walk(&[(KB, chunk / 2)])
        .walk_every(&[KB])
        .planes(&[(NI, 1)])
        .cubes(&[NB])
        .build();
    let chunks_level = levels[2].clone();
    let launcher = implied(&client, Partitioning::new(w.space(), levels), Form::Static);
    let s_op = w.s_op(&client, &launcher, scales, w.tile);

    partitioned_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w.a_op(&client, &launcher).arg(),
        w.b_op(&client, &launcher, None).arg(),
        s_op.arg(),
        w.c_op(&launcher, &c).arg(),
        launcher.partitioning_arg(),
        chunks_level,
        read,
        dtype,
    );
    w.check(&client, c, &format!("partition {scales:?}"));
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores_under_a_partition() {
    for read in [UnitRead::Shuffle, UnitRead::PlaneShared] {
        check_partitioned(TileScales::F32, read);
    }
}

#[test]
fn a_tile_ordered_weight_lands_on_the_tensor_cores_under_a_partition_with_byte_scales() {
    for read in [UnitRead::Shuffle, UnitRead::PlaneShared] {
        check_partitioned(TileScales::Ue4m3, read);
    }
}

/// `c = a · (b ⊗ s)` over a weight stored in tile order, a plane's step one block of the
/// contraction deep: two instructions under one scale, the step's weight scaled as one window.
#[cube(launch)]
fn stepped_scaled_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    for cube in space {
        let (a_cube, b_cube) = (a.at(&cube), b.at(&cube));
        let (scale_cube, c_cube) = (scale.at(&cube), c.at(&cube));
        for plane in cube {
            let (a_plane, b_plane) = (a_cube.at(&plane), b_cube.at(&plane));
            let (scale_plane, c_plane) = (scale_cube.at(&plane), c_cube.at(&plane));
            let sum =
                c_plane.accumulator::<E>(&a_plane, &b_plane, Instruction::Cmma, Semiring::SUM_PROD);
            for step in plane {
                let mut sum_step = sum.at(&step);
                sum_step.mma(
                    &a_plane.at(&step),
                    &b_plane.at(&step).mul(&scale_plane.at(&step)),
                );
            }
            sum.drained_into(&c_plane);
        }
    }
}

/// **A tile-ordered weight read two words a load lands on the tensor cores a block at a time.**
/// One load is a column's whole block, twice an instruction's depth: the step lands its window
/// once, a whole load at a time, and its two instructions read the landing.
fn check_stepped(scales: TileScales) {
    let (rows, n_tiles, k_tiles, fragment) = (16, 2, 4, 8);
    let w = TileOrdered::new(rows, n_tiles, k_tiles);
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let c = TestInput::builder(client.clone(), shape![rows, w.cols()])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // Leaf up: the instruction; the grid of fragments a plane holds; the block's two instruction
    // depths; the blocks, a step each; the planes; the cubes.
    let levels = Levels::leaf(&[(M, fragment), (NI, fragment), (KI, fragment)])
        .walk(&[(M, rows / fragment), (NI, w.tile / fragment)])
        .walk(&[(KI, w.tile / fragment)])
        .walk_every(&[KB])
        .planes(&[(NI, 1)])
        .cubes(&[NB])
        .build();
    let launcher = implied(&client, Partitioning::new(w.space(), levels), Form::Static);
    let s_op = w.s_op(&client, &launcher, scales, w.tile);

    stepped_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w.a_op(&client, &launcher).arg(),
        w.b_op(&client, &launcher, Some(w.tile)).arg(),
        s_op.arg(),
        w.c_op(&launcher, &c).arg(),
        launcher.partitioning_arg(),
        dtype,
    );
    w.check(&client, c, &format!("stepped {scales:?}"));
}

/// How a decoded copy reaches its output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum CopiedThrough {
    /// Straight from the weight.
    Directly,
    /// Straight from the weight, a block of `K` at a time.
    ABlockAtATime,
    /// Through a stage the copy fills a block of `K` at a time, each stage copied out as it lies.
    AStage,
}

/// A tile-ordered weight copied decoded: its words under their scales, into a dense output over
/// the weight's own axes, one cube a tile column, reaching it as `through` says.
#[cube(launch)]
fn decoded_tile_order_copy(
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] through: CopiedThrough,
) {
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let out = out.tile(&space);
    for cube in space {
        let decoded = b.at(&cube).mul(&scale.at(&cube));
        let mut out_cube = out.at(&cube);
        match comptime!(through) {
            CopiedThrough::Directly => out_cube.dequantize_from(&decoded),
            CopiedThrough::ABlockAtATime => {
                for block in cube.walk() {
                    let mut out_block = out_cube.at(&block);
                    out_block.dequantize_from(&decoded.at(&block));
                }
            }
            CopiedThrough::AStage => {
                let run = cube.walk();
                let mut stages =
                    Stages::smem_single(&run, &decoded, StageStorage::Strided, 1usize, None);
                stages.pipelined(run, |slot, stage| {
                    let mut out_stage = out_cube.at(stage);
                    slot.consume(|b_stage| {
                        out_stage.copy_from(b_stage);
                    });
                });
            }
        }
    }
}

/// **A tile-ordered weight copies decoded.** Every value lands as its `e2m1` code times its
/// column's scale, at its own coordinates: what a stage filled from the scaled weight holds.
#[test]
fn a_tile_ordered_weight_copies_decoded() {
    check_decoded_tile_order_copy(CopiedThrough::Directly);
}

#[test]
fn a_tile_ordered_weight_copies_decoded_a_block_at_a_time() {
    check_decoded_tile_order_copy(CopiedThrough::ABlockAtATime);
}

/// The same through a stage filled a block at a time: the stage holds what the copy decoded.
#[test]
fn a_tile_ordered_weight_copies_decoded_through_a_stage() {
    check_decoded_tile_order_copy(CopiedThrough::AStage);
}

fn check_decoded_tile_order_copy(through: CopiedThrough) {
    let w = TileOrdered::new(1, 2, 4);
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let space = Space::new(&[(NB, w.n_tiles), (KB, w.k_tiles), (NI, w.tile), (KI, w.tile)]);
    let levels = Levels::leaf(&[(KB, 1), (NI, w.tile), (KI, w.tile)])
        .walk_every(&[KB])
        .cubes(&[NB])
        .build();
    let launcher = implied(&client, Partitioning::new(space, levels), Form::Static);
    let shape = shape![w.n_tiles, w.k_tiles, w.tile, w.tile];
    let out = TestInput::builder(client.clone(), shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let s_op = w.s_op(&client, &launcher, TileScales::Ue4m3, w.tile);
    decoded_tile_order_copy::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w.b_op(&client, &launcher, Some(w.tile)).arg(),
        s_op.arg(),
        launcher
            .arg(out.clone().binding(), dtype)
            .axes(&[NB, KB, NI, KI])
            .vectorize(w.tile)
            .build()
            .unwrap()
            .arg(),
        launcher.partitioning_arg(),
        through,
    );
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for nb in 0..w.n_tiles {
        for kb in 0..w.k_tiles {
            for ni in 0..w.tile {
                for ki in 0..w.tile {
                    let want = e2m1::from_bits(w.code(nb, kb, ni, ki) as u8).to_f32()
                        * w.scale_at(nb, kb, ni);
                    let have = got.get_f32(&[nb, kb, ni, ki]);
                    assert_eq!(
                        have, want,
                        "{through:?} decoded at ({nb}, {kb}, {ni}, {ki})"
                    );
                }
            }
        }
    }
}

/// Where a staged weight's words are unpacked under their scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Unpacked {
    /// Each plane lands its window of the staged words, scaled.
    ByEachPlane,
    /// The copy filling the stage decodes the words, and the planes read the decoded stage.
    ByTheFill,
}

/// [`stepped_scaled_matmul`] with each cube staging the activation and the weight a block at a
/// time, held as `storage` says, the weight unpacked where `unpacked` says and dequantized at
/// `decoded` for the instruction, summing in `A`: a stage the fill decodes holds `decoded`, and
/// words staged as they are stored are dequantized at it by each plane.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn staged_stepped_scaled_matmul<A: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] unpacked: Unpacked,
    #[comptime] storage: StageStorage,
    #[comptime] decoded: ElemType,
    #[define(A)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let c = c.tile(&space);
    for cube in space {
        let (a_cube, b_cube) = (a.at(&cube), b.at(&cube));
        let (scale_cube, c_cube) = (scale.at(&cube), c.at(&cube));
        let sum = c_cube.accumulator::<A>(&a_cube, &b_cube, Instruction::Cmma, Semiring::SUM_PROD);
        let run = cube.walk();
        let staged = match comptime!(unpacked) {
            Unpacked::ByEachPlane => b_cube.clone(),
            Unpacked::ByTheFill => b_cube.mul(&scale_cube),
        };
        let mut stages = Stages::smem(
            &run,
            &a_cube,
            &staged,
            comptime!(storage.clone()),
            1usize,
            comptime!(Some(decoded)),
        );
        stages.walk(run, Prefetch::InSlots, |slot, stage| {
            let sum_stage = sum.at(stage);
            let scale_stage = scale_cube.at(stage);
            slot.consume(|a_stage, b_stage| {
                for plane in stage {
                    let mut sum_plane = sum_stage.at(&plane);
                    let b_plane = match comptime!(unpacked) {
                        Unpacked::ByEachPlane => b_stage.at(&plane).mul(&scale_stage.at(&plane)),
                        Unpacked::ByTheFill => b_stage.at(&plane),
                    };
                    let b_plane = sum_plane.dequantized(&b_plane, decoded);
                    sum_plane.mma(&a_stage.at(&plane), &b_plane);
                }
            });
        });
        sum.drained_into(&c_cube);
    }
}

/// The planes a cube splits its box between, along its rows and its columns.
#[derive(Clone, Copy, Debug)]
struct PlaneGrid {
    rows: usize,
    columns: usize,
}

impl PlaneGrid {
    const ONE: Self = Self {
        rows: 1,
        columns: 1,
    };
}

/// How a cube's stages hold their cells.
#[derive(Clone, Copy, Debug)]
enum StagesHeld {
    /// Row-major.
    Strided,
    /// In blocks of one instruction, each row padded by a chunk, as a stage `cmma` reads.
    InPaddedBlocks,
}

impl StagesHeld {
    /// The storage, its blocks `fragment` wide along every axis the instruction spans.
    fn storage(self, fragment: usize) -> StageStorage {
        match self {
            Self::Strided => StageStorage::Strided,
            Self::InPaddedBlocks => StageStorage::Tiled {
                block: vec![
                    (M, fragment),
                    (NB, 1),
                    (NI, fragment),
                    (KB, 1),
                    (KI, fragment),
                ],
                chunks: RowChunks::Padded,
            },
        }
    }
}

/// **A tile-ordered weight staged as its words lands a step at a time.** Each cube stages the
/// activation, in `elem`, and the weight's words a block deep; each of its planes lands its
/// window of the staged words, scaled, once a block, and runs the block's two instructions out of
/// the landing, summing in `f32`. The cube's planes split its rows and its columns as `planes`
/// says: planes on the same columns land the same window of the weight, each in a landing of its
/// own, and planes on the same rows read the same window of the activation.
fn check_staged_stepped(
    scales: TileScales,
    elem: ElemType,
    planes: PlaneGrid,
    unpacked: Unpacked,
    stages: StagesHeld,
) {
    let (plane_rows, n_tiles, k_tiles, fragment) = (16, 2, 4, 8);
    let rows = plane_rows * planes.rows;
    let w = TileOrdered::new(rows, n_tiles, k_tiles);
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    if !require_cmma_8x8x8(&client, elem, dtype) {
        return;
    }
    let c = TestInput::builder(client.clone(), shape![rows, w.cols()])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // Leaf up: the instruction; the grid of fragments a plane holds; the block's two instruction
    // depths; the planes; the blocks a cube stages, one a stage; the cubes.
    let levels = Levels::leaf(&[(M, fragment), (NI, fragment), (KI, fragment)])
        .walk(&[
            (M, plane_rows / fragment),
            (NI, w.tile / fragment / planes.columns),
        ])
        .walk(&[(KI, w.tile / fragment)])
        .planes(&[(M, planes.rows), (NI, planes.columns)])
        .walk_every(&[KB])
        .cubes(&[NB])
        .build();
    let launcher = implied(&client, Partitioning::new(w.space(), levels), Form::Static);
    // A plane holds its share of a tile's columns, a line of scales at a time.
    let s_op = w.s_op(&client, &launcher, scales, w.tile / planes.columns);

    staged_stepped_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        w.a_op_blocked(&client, &launcher, elem).arg(),
        w.b_op(&client, &launcher, Some(w.tile)).arg(),
        s_op.arg(),
        w.c_op(&launcher, &c).arg(),
        launcher.partitioning_arg(),
        unpacked,
        stages.storage(fragment),
        elem,
        dtype,
    );
    w.check(
        &client,
        c,
        &format!("staged stepped {scales:?} {elem:?} {unpacked:?} {stages:?}"),
    );
}

#[test]
fn a_tile_ordered_weight_staged_as_its_words_lands_a_step_at_a_time() {
    check_staged_stepped(
        TileScales::F32,
        f32::elem_type_native(),
        PlaneGrid::ONE,
        Unpacked::ByEachPlane,
        StagesHeld::Strided,
    );
}

#[test]
fn a_tile_ordered_weight_staged_as_its_words_lands_a_step_at_a_time_under_byte_scales() {
    check_staged_stepped(
        TileScales::Ue4m3,
        f32::elem_type_native(),
        PlaneGrid::ONE,
        Unpacked::ByEachPlane,
        StagesHeld::Strided,
    );
}

/// The same in halves: a half lands an `e2m1` code a lift short of its value, and the lift times
/// a block scale is past a half's range, so the landing applies the factor wider than the half it
/// writes.
#[test]
fn a_tile_ordered_weight_staged_as_its_words_lands_a_step_at_a_time_in_halves() {
    check_staged_stepped(
        TileScales::Ue4m3,
        f16::elem_type_native(),
        PlaneGrid::ONE,
        Unpacked::ByEachPlane,
        StagesHeld::Strided,
    );
}

/// The same with two planes splitting the rows: both land the same window of the weight, each
/// in a landing of its own.
#[test]
fn a_tile_ordered_weight_staged_as_its_words_lands_a_step_at_a_time_for_every_plane_sharing_it() {
    check_staged_stepped(
        TileScales::Ue4m3,
        f16::elem_type_native(),
        PlaneGrid {
            rows: 2,
            columns: 1,
        },
        Unpacked::ByEachPlane,
        StagesHeld::Strided,
    );
}

/// The same with two planes splitting the columns: each lands its own window of the weight under
/// its own columns' scales.
#[test]
fn a_tile_ordered_weight_staged_as_its_words_lands_a_step_at_a_time_a_window_a_plane() {
    check_staged_stepped(
        TileScales::Ue4m3,
        f16::elem_type_native(),
        PlaneGrid {
            rows: 1,
            columns: 2,
        },
        Unpacked::ByEachPlane,
        StagesHeld::Strided,
    );
}

/// The same weight decoded by the copy filling each stage instead: the stage holds the weight's
/// values under their scales at the activation's element, as the kernel names it, and the planes
/// contract the decoded stage as they would a float one.
#[test]
fn a_tile_ordered_weight_decoded_by_the_stage_fill_contracts_as_its_values() {
    check_staged_stepped(
        TileScales::Ue4m3,
        f16::elem_type_native(),
        PlaneGrid::ONE,
        Unpacked::ByTheFill,
        StagesHeld::Strided,
    );
}

/// The same with two planes splitting the rows, both reading the one stage the cube decoded.
#[test]
fn a_tile_ordered_weight_decoded_by_the_stage_fill_serves_every_plane_sharing_it() {
    check_staged_stepped(
        TileScales::Ue4m3,
        f16::elem_type_native(),
        PlaneGrid {
            rows: 2,
            columns: 1,
        },
        Unpacked::ByTheFill,
        StagesHeld::Strided,
    );
}

/// The same into stages held in padded blocks: the copy decodes each value into the place the
/// stage's layout gives it, whatever that layout is.
#[test]
fn a_tile_ordered_weight_decoded_by_the_stage_fill_lands_in_padded_blocks() {
    check_staged_stepped(
        TileScales::Ue4m3,
        f16::elem_type_native(),
        PlaneGrid {
            rows: 2,
            columns: 2,
        },
        Unpacked::ByTheFill,
        StagesHeld::InPaddedBlocks,
    );
}

#[test]
fn a_tile_ordered_weight_read_a_block_a_load_lands_a_step_at_a_time() {
    check_stepped(TileScales::F32);
}

#[test]
fn a_tile_ordered_weight_read_a_block_a_load_lands_a_step_at_a_time_under_byte_scales() {
    check_stepped(TileScales::Ue4m3);
}

/// `out = a ⊗ s`, the decode stated where the kernel copies: straight into `out`, or into a
/// shared stage first, which is how a kernel keeps a stage of decoded values.
#[cube(launch)]
fn scaled_copy<E: Numeric>(
    a: &TileArg<'_>,
    scale: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] staged: bool,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let scale = scale.tile(&space);
    let mut out = out.tile(&space);
    if comptime!(staged) {
        let mut stage = Tile::shared::<E>(comptime!(space.space().clone()), StageStorage::Strided);
        stage.dequantize_from(&a.mul(&scale));
        sync_cube();
        out.copy_from(&stage);
    } else {
        out.dequantize_from(&a.mul(&scale));
    }
}

fn check_scaled_copy(staged: bool) {
    let (rows, block, blocks) = (4, 8, 4);
    let depth = block * blocks;
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth).map(|i| (i % 5) as f32 - 2.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    // A stage is dense over its axes, one dim each; copied back out, it lands in a buffer laid
    // out the same way. A direct copy writes the values' own layout.
    let out_shape = match staged {
        true => shape![rows, blocks, block],
        false => shape![rows, depth],
    };
    let out = TestInput::builder(client.clone(), out_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (KB, blocks), (KI, block)]).build(),
        ),
        Form::Static,
    );
    let projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    scaled_copy::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(projection.clone()),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(projection.scales_per(KB)),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            dtype,
            1,
            match staged {
                true => TileSpec::direct(&[M, KB, KI]),
                false => TileSpec::new(projection),
            },
        ),
        launcher.partitioning_arg(),
        staged,
        dtype,
    );
    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for m in 0..rows {
        for k in 0..depth {
            let want = a[m * depth + k] * s[m * blocks + k / block];
            let have = match staged {
                true => got.get_f32(&[m, k / block, k % block]),
                false => got.get_f32(&[m, k]),
            };
            assert!(
                (have - want).abs() < 1e-6,
                "at ({m}, {k}): got {have}, want {want}"
            );
        }
    }
}

/// A copy of scaled values decodes them where the kernel wrote the copy.
#[test]
fn a_copy_of_scaled_values_decodes_them() {
    check_scaled_copy(false);
}

/// The same decode into a shared stage: the stage holds the decoded values, because the kernel
/// said so at the copy that filled it.
#[test]
fn a_stage_filled_from_scaled_values_holds_them_decoded() {
    check_scaled_copy(true);
}

/// `c = (a ⊗ s) · b` with `a` staged **decoded**: the stages are filled from `a.mul(&s)`, so the
/// `mul` comes before the stage and the copy filling each slot is where `a` decodes; the leaf reads
/// plain values.
#[cube(launch)]
fn staged_scaled_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    scale: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] depth: usize,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let scale = scale.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem(
        &walk,
        &a.mul(&scale),
        &b,
        StageStorage::Strided,
        depth,
        None,
    );
    stages.pipelined(walk, |slot, region| {
        let mut c_r = c
            .at(region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        slot.consume(|a_s, b_s| {
            c_r.mma(a_s, b_s);
        });
    });
}

fn check_staged_scaled_matmul(depth: usize) {
    let (rows, cols, block, blocks) = (4, 4, 8, 4);
    let depth_k = block * blocks;

    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..rows * depth_k).map(|i| (i % 5) as f32 - 2.0).collect();
    let b: Vec<f32> = (0..depth_k * cols).map(|i| (i % 7) as f32 - 3.0).collect();
    let s: Vec<f32> = (0..rows * blocks).map(|i| (i as f32 + 1.0) / 2.0).collect();
    let (a_t, _) = TestInput::builder(client.clone(), shape![rows, depth_k])
        .dtype(dtype)
        .custom(a.clone())
        .generate_with_f32_host_data();
    let (b_t, _) = TestInput::builder(client.clone(), shape![depth_k, cols])
        .dtype(dtype)
        .custom(b.clone())
        .generate_with_f32_host_data();
    let (s_t, _) = TestInput::builder(client.clone(), shape![rows, blocks])
        .dtype(dtype)
        .custom(s.clone())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // One block of `K` a stage, walked.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols), (KB, blocks), (KI, block)]),
            Levels::leaf(&[(M, rows), (N, cols), (KB, 1), (KI, block)])
                .walk_every(&[M, N, KB, KI])
                .build(),
        ),
        Form::Static,
    );
    let a_projection = Projection::new(
        &[M, KB, KI],
        &[
            PhysicalAxisMap::of(M),
            PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
        ],
    );
    staged_scaled_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(a_projection.clone()),
        ),
        TileArgLaunch::new(
            b_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, block), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            s_t.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(a_projection.scales_per(KB)),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        depth,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    for m in 0..rows {
        for n in 0..cols {
            let want: f32 = (0..depth_k)
                .map(|k| a[m * depth_k + k] * s[m * blocks + k / block] * b[k * cols + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-3,
                "depth {depth}, at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// Stages filled from scaled values hold them decoded, one slot or two in flight.
#[test]
fn stages_filled_from_scaled_values_contract_them_decoded() {
    check_staged_scaled_matmul(1);
    check_staged_scaled_matmul(2);
}
