//! A plane group's warpgroup MMA over stages the TMA engine lands: two groups of four planes a
//! cube, each one `64 × 128` accumulator, its MMAs reading both operands K-major straight out of
//! the stage, beside a plane that fills the stages and frees each once the MMAs that read it are
//! done ([`Stages::consume_async`]). Checked against the host product, one stage in flight and
//! two.
//!
//! A device without the warpgroup MMA or the TMA engine skips: Hopper alone has both.

use crate::support::ValidationResult;
use cubecl::{
    features::{Tma, WgmmaConfig, WgmmaElems},
    prelude::*,
    zspace::{shape, strides},
};
use cubecl_tile::launch::TmaBox;
use cubecl_tile::space::ComputeScope;
use cubecl_tile::stage::{AsyncStage, AsyncStageExpand, Role, RowChunks};
use cubecl_tile::*;

use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// One plane group's tile and one instruction step of `K`.
const GROUP: (usize, usize) = (64, 128);
const STEP: usize = 16;
/// Steps of `K` a stage: one 128-byte swizzle span of `f16`.
const STEPS: usize = 4;
/// Plane groups a cube, along `m`.
const GROUPS: usize = 2;

/// Each group's MMAs into its window of `sum`, every step of the stage.
#[derive(CubeType)]
struct GroupMma {
    sum: Tile,
}

#[cube]
impl AsyncStage for GroupMma {
    fn issue(&self, region: &Region, lhs: &Tile, rhs: &Tile) -> Pending<()> {
        let sum = self.sum.at(region);
        for group in region {
            let (sum, lhs, rhs) = (sum.at(&group), lhs.at(&group), rhs.at(&group));
            for step in group.walk().unrolled() {
                let mut acc = sum.at(&step);
                acc.mma(&lhs.at(&step), &rhs.at(&step));
            }
        }
        self.sum.commit()
    }

    fn done(&self) -> Pending<()> {
        self.sum.commit()
    }
}

/// `c = a · bᵀ`, each cube's box walked a stage of `K` at a time.
#[cube(launch)]
fn wgmma_walk(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in space.over(&cubes) {
        let (a, b, c) = (a.at(&cube), b.at(&cube), c.at(&cube));
        let walk = cube.walk();
        let mut stages = Stages::smem(&walk, &a, &b, comptime!(storage.clone()), depth, None);
        let total = walk.total();
        let laps = (total + comptime!(depth - 1)) / comptime!(depth);
        match stages.role() {
            Role::Fill => {
                for lap in 0..laps {
                    #[unroll]
                    for slot in 0..depth {
                        let index = lap * comptime!(depth) + slot;
                        if index < total {
                            stages.fill(slot, &walk.region(index));
                        }
                    }
                }
            }
            Role::Compute => {
                let mut sum =
                    c.accumulator::<f32>(&a, &b, comptime!(Instruction::Wgmma), Semiring::SUM_PROD);
                sum.zero();
                let mma = GroupMma { sum: sum.clone() };
                stages.consume_async::<GroupMma>(&walk, &mma);
                sum.drained_into(&c);
            }
        }
    }
}

/// The partitioning of an `m × n × k` product: the instruction's step, the group's one
/// accumulator, the steps of a stage, two plane groups along `m`, the stages along `K` beside one
/// plane that fills them, and the cubes.
fn partitioning(m: usize, n: usize, k: usize) -> Partitioning {
    let (gm, gn) = GROUP;
    Partitioning::new(
        Space::new(&[(M, m), (N, n), (K, k)]),
        Levels::leaf(&[(M, gm), (N, gn), (K, STEP)])
            .walk(&[(M, 1), (N, 1)])
            .walk(&[(K, STEPS)])
            .plane_groups(4, &[(M, GROUPS)])
            .walk_every(&[K])
            .filled_by(1)
            .cubes(&[M, N])
            .build(),
    )
}

/// The stage both operands land in: one group's tile along `m` and `n`, the stage's whole `K`,
/// rows swizzled as the descriptor swizzles them.
fn storage() -> StageStorage {
    let (gm, gn) = GROUP;
    StageStorage::Tiled {
        block: vec![(M, gm), (N, gn), (K, STEP * STEPS)],
        chunks: RowChunks::Swizzled,
    }
}

/// Whether this device runs a `64 × 128` warpgroup MMA of `f16` into `f32` and lands its stages
/// by TMA; a test without them is skipped, not passed.
fn runs_wgmma(client: &Client) -> bool {
    let props = client.properties();
    let f16 = half::f16::elem_type_native();
    let elems = WgmmaElems {
        a: f16,
        b: f16,
        cd: f32::elem_type_native(),
    };
    let shape = cubecl::ir::types::MatrixShape {
        m: 64,
        n: GROUP.1,
        k: STEP,
    };
    let wgmma = props
        .features
        .matmul
        .wgmma
        .iter()
        .any(|config: &WgmmaConfig| config.matches(elems, shape));
    if wgmma && props.features.tma.contains(Tma::Base) {
        return true;
    }
    ValidationResult::Skipped("the device has no warpgroup MMA or no TMA engine".to_string())
        .enforce();
    false
}

/// A row-major `rows × cols` `f16` matrix as a TMA tensor map of `stage_rows × stage_cols` boxes,
/// swizzled as `storage` lands them.
fn tma_matrix(
    client: &Client,
    values: &[half::f16],
    (rows, cols): (usize, usize),
    (stage_rows, stage_cols): (usize, usize),
    axes: [Axis; 2],
    swizzle: TensorMapSwizzle,
) -> TileArgLaunch {
    let handle = client.create_from_slice(half::f16::as_bytes(values));
    let tensor = unsafe {
        TensorArg::from_raw_parts(
            handle,
            strides![rows * cols, cols, 1],
            shape![1, rows, cols],
        )
    };
    let map = TensorMapArg::new(
        TiledArgs {
            tile_size: shape![1, stage_rows, stage_cols],
        },
        tensor,
        half::f16::elem_type_native(),
    )
    .with_swizzle(swizzle);
    TileArgLaunch::tensor_map(
        map,
        &axes,
        TmaBox {
            rows: rows as u32,
            cols: cols as u32,
            batch: None,
            transposed: false,
        },
    )
}

fn check_wgmma_walk(m: usize, n: usize, k: usize, depth: usize) {
    let client = cubecl::test_device().client();
    if !runs_wgmma(&client) {
        return;
    }
    let lhs: Vec<f32> = (0..m * k)
        .map(|i| ((i / k * 3 + i % k) % 7) as f32 - 3.0)
        .collect();
    let rhs: Vec<f32> = (0..n * k)
        .map(|i| ((i / k * 5 + i % k) % 5) as f32 - 2.0)
        .collect();
    let half = |values: &[f32]| -> Vec<half::f16> {
        values.iter().map(|&v| half::f16::from_f32(v)).collect()
    };

    let launcher = implied(&client, partitioning(m, n, k), Form::Static);
    let storage = storage();
    let stage = |rows: Axis| {
        let space = launcher.partitioning().box_of(ComputeScope::Cube);
        Space::new(&[(rows, space.extent(rows)), (K, STEP * STEPS)])
    };
    let swizzle = |rows: Axis| {
        storage
            .tma_swizzle(&stage(rows), 1, 16)
            .expect("a box lands the stage")
    };
    let (box_m, box_n) = (stage(M).extent(M), stage(N).extent(N));
    let a = tma_matrix(
        &client,
        &half(&lhs),
        (m, k),
        (box_m, STEP * STEPS),
        [M, K],
        swizzle(M),
    );
    let b = tma_matrix(
        &client,
        &half(&rhs),
        (n, k),
        (box_n, STEP * STEPS),
        [N, K],
        swizzle(N),
    );
    let out = client.empty(m * n * size_of::<f32>());
    let c = unsafe { TensorArg::from_raw_parts(out.clone(), strides![n, 1], shape![m, n]) };

    wgmma_walk::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a,
        b,
        TileArgLaunch::new(c, f32::elem_type_native(), 1, TileSpec::direct(&[M, N])),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        storage,
        depth,
    );

    let bytes = client.read_one(out).expect("the product reads back");
    let got = f32::from_bytes(&bytes);
    // A launch whose kernel failed to compile reads back nothing.
    assert_eq!(got.len(), m * n, "the kernel wrote no product");
    for row in 0..m {
        for col in 0..n {
            let want: f32 = (0..k).map(|p| lhs[row * k + p] * rhs[col * k + p]).sum();
            assert_eq!(
                got[row * n + col],
                want,
                "depth {depth}: c[{row}][{col}] of {m}x{n}x{k}"
            );
        }
    }
}

/// Two stages in their slots: each slot freed a stage after its MMAs were issued.
#[test]
fn a_warpgroup_mma_walk_frees_each_stage_a_stage_late() {
    check_wgmma_walk(256, 256, 256, 2);
}

/// One slot: the MMAs that read it are waited for before it is refilled.
#[test]
fn a_warpgroup_mma_walk_over_one_slot_waits_at_once() {
    check_wgmma_walk(128, 128, 192, 1);
}
