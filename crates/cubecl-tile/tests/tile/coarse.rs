//! A **coarse** operand: one value per block of an axis, rather than one per element.
//!
//! The shape a per-block quantization scale has, `s[m, k / block]` beside `v[m, k]`, and the one
//! capability the explicit-scales design needs, since there the scales are a real kernel operand.
//! Proven with plain floats, deliberately: a mechanism only the quant machinery can use is not one.
//!
//! The spelling is a rational [`Projection`]: `⌊k / BLOCK⌋`, the same floor the resample
//! mapping already rides, so a coarse operand is a gather like any other, and nothing about it
//! is quantization's.
//!
//! The probe is a contraction, not a copy, because the read is what the design needs: a scale is
//! consumed where the values are, never staged into the shape of its own expansion. [`Tile::copy`]
//! refuses this outright: a compacted stage fill needs source and destination on one projection.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::shape};
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

const ROWS: usize = 4;
const COLS: usize = 4;
const DEPTH: usize = 32;
/// Contracted values per coarse value: the quantization block, in the shape this stands in for.
const BLOCK: usize = 8;
const BLOCKS: usize = DEPTH / BLOCK;
/// The leaf's register block.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `c = a · b` with `a` coarse along the contracted axis: one value per block of `K`, read by
/// every `k` the block covers.
#[cube(launch)]
fn coarse_lhs_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    c.zero();
    for region in space.over(&level) {
        let mut c_region = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        c_region.mma(&a.at(&region), &b.at(&region));
    }
}

/// `⌊k / BLOCK⌋` on the contracted axis, the row addressed as it stands: the coarse operand's
/// whole declaration.
fn coarse_spec() -> TileSpec {
    TileSpec::new(Projection::new(
        &[M, K],
        &[PhysicalAxisMap::of(M), PhysicalAxisMap::of(K).over(BLOCK)],
    ))
}

/// One level, cutting `K` at `cut` so a walk that cuts *at* the block, finer, and coarser are
/// all expressible.
fn space(cut: usize) -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, ROWS), (N, COLS), (K, DEPTH)]),
            Levels::leaf(&[(M, ROWS), (N, COLS), (K, cut)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    )
}

/// Distinct per `(m, block)` and not integers, so an off-by-one block index cannot pass.
fn coarse_data() -> Vec<f32> {
    (0..ROWS * BLOCKS).map(|i| i as f32 + 0.5).collect()
}

/// Distinct per `(k, n)`, and small enough that the sum stays exact in `f32`.
fn rhs_data() -> Vec<f32> {
    (0..DEPTH * COLS).map(|i| (i % 7) as f32 - 3.0).collect()
}

/// Launch [`coarse_lhs_matmul`] over `space` and return `c`.
fn run(launcher: Launcher) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let (a, _) = TestInput::builder(client.clone(), shape![ROWS, BLOCKS])
        .dtype(dtype)
        .custom(coarse_data())
        .generate_with_f32_host_data();
    let (b, _) = TestInput::builder(client.clone(), shape![DEPTH, COLS])
        .dtype(dtype)
        .custom(rhs_data())
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![ROWS, COLS])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    coarse_lhs_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(a.binding().into_tensor_arg(), dtype, 1, coarse_spec()),
        TileArgLaunch::new(
            b.binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    HostData::from_tensor_handle(&client, c, HostDataType::F32)
}

/// Every `k` of a block contracted against that block's one coarse value.
fn assert_contracted(got: &HostData) {
    let (a, b) = (coarse_data(), rhs_data());
    for m in 0..ROWS {
        for n in 0..COLS {
            let want: f32 = (0..DEPTH)
                .map(|k| a[m * BLOCKS + k / BLOCK] * b[k * COLS + n])
                .sum();
            let have = got.get_f32(&[m, n]);
            assert!(
                (have - want).abs() < 1e-4,
                "at ({m}, {n}): got {have}, want {want}"
            );
        }
    }
}

/// The walk cuts `K` exactly at the block, which is the alignment the explicit-scales design
/// asks a routine for: one coarse value per region.
#[test]
fn a_coarse_operand_contracts_over_its_block() {
    assert_contracted(&run(space(BLOCK)));
}

/// A cut finer than the block: several regions in a row read the same coarse value, which is
/// omission-is-invariance arriving through the floor rather than through a missing axis.
#[test]
fn a_cut_finer_than_the_block_reads_the_same_value_across_regions() {
    assert_contracted(&run(space(BLOCK / 2)));
}

/// A cut coarser than the block: one region spans several coarse values, so the read addresses
/// them within the region rather than once per region.
#[test]
fn a_cut_coarser_than_the_block_addresses_each_value_in_the_region() {
    assert_contracted(&run(space(BLOCK * 2)));
}
