//! Splitting the contraction across cubes, and putting the pieces back together.
//!
//! A cube that does not do the whole contraction holds a slice of every output cell it touches,
//! and the slices have to be combined. This file is the combine that needs nothing new: the split
//! is spelled as an *axis*.
//!
//! `K` is declared as two axes, `(KB, KI)`, addressing one physical `K` through
//! [`PhysicalAxisMap::disjoint`] exactly as a quantization block does. `KB` counts the splits and
//! rides the cubes; `KI` is the position inside one split and is what the contraction walks.
//!
//! The output is bound over `[KB, M, N]`, so it *spans* the axis being split, and the whole thing
//! is a batched matmul whose batch is the split index: no cube shares a cell with any other and
//! nothing is partial. A second pass reduces the `KB` axis away.
//!
//! Two kernels and an extra buffer, but the engine is untouched, which is what makes this the
//! reference the in-kernel combines are measured against: it is the same arithmetic, cut the same
//! way, differing only in where the pieces are added up.
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    CubeCount, CubeDim,
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    zspace::shape,
};

use super::{Form, implied};
use cubecl_tile::launch::AccumulateArg;
use cubecl_tile::launch::AccumulateArgLaunch;
use cubecl_tile::layout::PhysicalAxisMap;
use cubecl_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
// `K` as two axes: `KB` counts the splits (one per cube), `KI` walks inside one.
const KB: Axis = Axis(2);
const KI: Axis = Axis(3);

/// The leaf's register block, held fixed across every kernel here.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// The split contraction: a batched matmul whose batch is the split index, writing one partial
/// per split. One region per cube, contracted whole at the leaf, so `mm` owns the init and
/// the partials buffer needs no zeroing.
#[cube(launch)]
fn split_partials(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    partials: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let partials = partials.tile(&space);
    for region in space.over(&level) {
        let mut partials_cube = partials
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        partials_cube.mm(&a.at(&region), &b.at(&region));
    }
}

/// The second pass: fold the split axis away.
#[cube(launch)]
fn reduce_splits(
    partials: &TileArg<'_>,
    out: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
) {
    let partials = partials.tile(&space);
    let out = out.tile(&space);
    for region in space.over(&level) {
        let mut out_cube = out.at(&region);
        out_cube.reduce_axis(&partials.at(&region), Monoid::Sum);
    }
}

/// The whole pipeline over `(m, n, k)` cut into `splits`: the partials, then the fold. Returns
/// both, since a wrong total and a wrong partial are different bugs.
fn run_split_contraction(m: usize, n: usize, k: usize, splits: usize) -> (HostData, HostData) {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    assert!(k.is_multiple_of(splits), "the test shapes divide evenly");
    let inside = k / splits;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    // Poisoned, not zeroed: `mm` states `c = a·b`, so nothing here may count.
    let partials = TestInput::builder(client.clone(), shape![splits, m, n])
        .dtype(dtype)
        .uniform(4242, 10., 100.)
        .generate_without_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .uniform(4243, 10., 100.)
        .generate_without_host_data();

    // One split per cube, the whole output tile in each: the split is the only thing on the grid.
    let split_space = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (KB, splits), (KI, inside)]),
            Levels::leaf(&[(KB, 1)]).cubes(&[KB]).build(),
        ),
        Form::Static,
    );

    // `a` is `[M, K]` and `b` is `[K, N]` in memory: one physical `K` dim each, addressed by the
    // two logical axes. `inside` is `KB`'s stride through it, `1` is `KI`'s.
    split_partials::launch(
        &client,
        split_space.cube_count(),
        split_space.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[M, KB, KI],
                &[
                    PhysicalAxisMap::of(M),
                    PhysicalAxisMap::disjoint(&[(KB, inside), (KI, 1)]),
                ],
            )),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::new(Projection::new(
                &[KB, KI, N],
                &[
                    PhysicalAxisMap::disjoint(&[(KB, inside), (KI, 1)]),
                    PhysicalAxisMap::of(N),
                ],
            )),
        ),
        TileArgLaunch::new(
            partials.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[KB, M, N]),
        ),
        split_space.partitioning_arg(),
        split_space.partitioning().level(0),
    );

    let reduce_space = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (KB, splits)]),
            Levels::leaf(&[(M, 1)]).cubes(&[M]).build(),
        ),
        Form::Static,
    );

    reduce_splits::launch(
        &client,
        reduce_space.cube_count(),
        reduce_space.cube_dim(),
        TileArgLaunch::new(
            partials.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[KB, M, N]),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        reduce_space.partitioning_arg(),
        reduce_space.partitioning().level(0),
    );

    (
        HostData::from_tensor_handle(&client, partials, HostDataType::F32),
        HostData::from_tensor_handle(&client, out, HostDataType::F32),
    )
}

/// The reference: `a·b` over the whole `K`, and over one split's slice of it.
fn reference(m: usize, n: usize, k: usize, rows: std::ops::Range<usize>) -> Vec<f32> {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            rows.clone().map(|p| a(i, p) * b(p, j)).sum()
        })
        .collect()
}

/// Each cube contracts its own slice of `K` and writes it to its own plane of the partials
/// buffer, and the fold adds the planes up. Both halves are checked: a partials buffer that is
/// right and a total that is not would be a broken fold, and the reverse a broken split.
#[test]
fn a_split_contraction_sums_back_to_the_whole() {
    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);
    let inside = k / splits;
    let (partials, out) = run_split_contraction(m, n, k, splits);

    for s in 0..splits {
        let want = reference(m, n, k, s * inside..(s + 1) * inside);
        for i in 0..m {
            for j in 0..n {
                let have = partials.get_f32(&[s, i, j]);
                let want = want[i * n + j];
                assert!(
                    (have - want).abs() < 1e-3,
                    "split {s} at ({i}, {j}): got {have}, want {want}"
                );
            }
        }
    }

    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = out.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "total at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// A split of one is the unsplit contraction, and has to stay it: the degenerate case is what a
/// selection heuristic falls back to, so it must not be a different program's worth of bugs.
#[test]
fn a_split_of_one_is_the_whole_contraction() {
    let (m, n, k) = (4usize, 4usize, 16usize);
    let (_, out) = run_split_contraction(m, n, k, 1);

    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = out.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The precondition the in-kernel combine rests on: this device folds `f32` atomically, and
/// really does fold rather than race, across cubes that all target one cell.
///
/// Probed rather than assumed. `f32` atomic add is there on metal (native and through wgpu's MSL
/// path), CUDA from sm60, and the CPU runtime, but WGSL only has it behind
/// `SHADER_FLOAT32_ATOMIC`: a real fork, not a formality; nothing else here matters if it fails.
#[cube(launch)]
fn atomic_add_probe(out: &mut Tensor<Atomic<f32>>) {
    if UNIT_POS == 0 {
        out[0].fetch_add(1.0);
    }
}

#[test]
fn the_device_adds_floats_atomically_across_cubes() {
    let client = cubecl::test_device().client();
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
        return;
    }

    let cubes = 64u32;
    let out = TestInput::builder(client.clone(), shape![1])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    atomic_add_probe::launch(
        &client,
        CubeCount::new_1d(cubes),
        CubeDim::new_single(),
        out.clone().binding().into_tensor_arg(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    // Every cube added one. A lost update reads back low, never high.
    assert_eq!(got.get_f32(&[0]), cubes as f32);
}

// -- The in-kernel combine --------------------------------------------------
//
// The same split, without the second buffer and the second pass: `K` stays one axis, the cubes
// each take a slice, and the drain folds each cube's contribution into the output atomically.
// What the workspace pipeline above does in two kernels, this does in one, and the two must agree.

const K: Axis = Axis(4);

/// The output is bound as an atomic buffer and drained through a folding sink. The register
/// accumulator is not decoration: the contraction runs in registers and only the drain touches
/// the destination, which is the one shape a write-only fold admits.
#[cube(launch)]
fn atomic_split_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = out.tile(&space);
    // The accumulator mirrors the output's grid at this level: opened above the walk, one
    // fragment per region, drained once through the sink after it.
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Registers {
            config: REGISTER_BLOCK
        }),
        Semiring::SUM_PROD,
    );
    for region in space.over(&level) {
        let mut acc_region = acc.at(&region);
        acc_region.mma(&a.at(&region), &b.at(&region));
    }
    // Drained through the levels it was opened under, which is what puts a unit's block in front
    // of its own columns; every unit writes there, and one writes where they repeat.
    acc.drained_into(&c);
}

/// The same, with the columns distributed out to the plane's units: one level more, and the unit
/// level is walked like any other. A unit's block is then its own columns, which is what makes
/// every unit a writer on the drain.
#[cube(launch)]
fn atomic_split_matmul_by_unit<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] plane_units: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = out.tile(&space);
    // Opened above both walks, so it holds what one unit of one cube sums: its own columns
    // against that cube's slice of the contraction.
    let acc = c.accumulator::<E>(
        &a,
        &b,
        comptime!(Instruction::Registers {
            config: REGISTER_BLOCK
        }),
        Semiring::SUM_PROD,
    );
    for cube in space.over(&cubes) {
        for unit in cube.over(&plane_units) {
            let mut acc_unit = acc.at(&unit);
            acc_unit.mma(&a.at(&unit), &b.at(&unit));
        }
    }
    acc.drained_into(&c);
}

/// `a·b` with `K` distributed out over `splits` cubes, folded atomically into a zeroed output.
fn run_atomic_split_contraction(m: usize, n: usize, k: usize, splits: usize) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    // Zeroed, and this is where `mm`'s init went: the operation states `c = a·b`, so something
    // has to put the identity there, and under a split that something is the launch.
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(K, k / splits)]).cubes(&[K]).build(),
        ),
        Form::Static,
    );

    atomic_split_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// The in-kernel combine against the same contraction, unsplit: every cube folds its own slice of
/// `K` into the output and the sum is the whole.
#[test]
fn an_atomic_drain_sums_the_slices_back_together() {
    let client = cubecl::test_device().client();
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
        return;
    }

    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);
    let got = run_atomic_split_contraction(m, n, k, splits);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The two combines agree. Not a restatement of the test above: that one checks the arithmetic,
/// this one checks that the two ways of putting the slices back together are the same program to
/// the caller, which is what makes them interchangeable at launch.
#[test]
fn the_atomic_drain_agrees_with_the_workspace() {
    let client = cubecl::test_device().client();
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
        return;
    }

    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);
    let (_, workspace) = run_split_contraction(m, n, k, splits);
    let atomic = run_atomic_split_contraction(m, n, k, splits);
    for i in 0..m {
        for j in 0..n {
            let w = workspace.get_f32(&[i, j]);
            let a = atomic.get_f32(&[i, j]);
            assert!(
                (w - a).abs() < 1e-3,
                "at ({i}, {j}): workspace {w}, atomic {a}"
            );
        }
    }
}

/// The same fold with the units carrying cells of their own: `N` rides the plane's units while
/// `K` rides the cubes, so a unit owns its columns and a cube owns its slice of the contraction.
///
/// The control on the writer election. A fold from units that repeat each other's work has to be
/// made by one of them, and a fold from units that each hold their own cells by all of them: an
/// election that cannot tell them apart is wrong one way; "unit zero writes" would drop this half.
#[test]
fn an_atomic_drain_with_units_of_their_own() {
    let client = cubecl::test_device().client();
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
        return;
    }
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let dtype = f32::elem_type_native();

    let (m, k, splits, per_unit) = (4usize, 16usize, 4usize, 2usize);
    let n = plane_size * per_unit;

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(N, per_unit), (K, k / splits)])
                .units(&[(N, plane_size)])
                .cubes(&[K])
                .build(),
        ),
        Form::Static,
    );

    atomic_split_matmul_by_unit::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The same fold one scope down: `K` cut across the *planes* of a single cube. Planes share no
/// registers either, so each holds a slice of every cell and the drain folds them in, and the
/// election is per plane, so one unit of each folds its own plane's contribution.
///
/// One cube, so nothing here is a cube split at all: what is being checked is that the combine is
/// about instances that cannot meet in registers, not about cubes in particular.
#[test]
fn an_atomic_drain_sums_across_planes() {
    let client = cubecl::test_device().client();
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
        return;
    }
    let dtype = f32::elem_type_native();
    let (m, n, k, num_planes) = (4usize, 4usize, 16usize, 4usize);

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(K, k / num_planes)])
                .planes(&[(K, num_planes)])
                .build(),
        ),
        Form::Static,
    );

    atomic_split_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

/// The output contracted *in place*, with no register accumulator at all.
///
/// The verb is still `mm`, and it is still true: across all the cubes the operation is
/// `c = a·b`. What the split moves is the *init* it owns: a cell belongs to several cubes, so
/// none may seed it, and the buffer arrives holding the fold's identity: zeroed before the launch.
///
/// Every write is then a `+=` into a cell that already holds what the other cubes contracted, and
/// nothing is ever read back, because folding is itself the read-modify-write.
#[cube(launch)]
fn atomic_split_matmul_in_place<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    #[comptime] level: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = out.tile(&space);
    for region in space.over(&level) {
        let mut c_region = c
            .at(&region)
            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
        c_region.mm(&a.at(&region), &b.at(&region));
    }
}

#[test]
fn a_summing_output_contracts_in_place() {
    let client = cubecl::test_device().client();
    if !client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add)
    {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
        return;
    }
    let dtype = f32::elem_type_native();
    let (m, n, k, splits) = (4usize, 4usize, 16usize, 4usize);

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(K, k / splits)]).cubes(&[K]).build(),
        ),
        Form::Static,
    );

    atomic_split_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let want = reference(m, n, k, 0..k);
    for i in 0..m {
        for j in 0..n {
            let have = got.get_f32(&[i, j]);
            let want = want[i * n + j];
            assert!(
                (have - want).abs() < 1e-3,
                "at ({i}, {j}): got {have}, want {want}"
            );
        }
    }
}

// -- The in-kernel combine, tensor-core leaf --------------------------------------------------
//
// A cmma accumulator stores through its intrinsic, which replaces and elects no writer, so on its
// own it cannot drain into a store that folds. Opened with a scratch, the partition bounces each
// fragment through smem, its one door, and the units fold its cells atomically, as a block does.

/// [`atomic_split_matmul`]'s tensor-core twin: the cube's slice of `K` staged and contracted in
/// fragments, the accumulator opened with a scratch and drained through the folding sink.
#[cube(launch)]
fn atomic_split_cmma<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = out.tile(&space);
    for cube in space {
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        let c_cube = c.at(&cube);
        let acc = c_cube
            .accumulator::<E>(&a_cube, &b_cube, Instruction::Cmma, Semiring::SUM_PROD)
            .with_scratch(Scratch::OneTile);
        let walk = cube.walk();
        let mut stages = Stages::smem(
            &walk,
            &a_cube,
            &b_cube,
            comptime!(StageStorage::Strided),
            1usize,
            None,
        );
        stages.pipelined(walk, |slot, stage| {
            let mut acc_s = acc.at(stage);
            slot.consume(|a_s, b_s| {
                acc_s.mma(a_s, b_s);
            });
        });
        acc.drained_into(&c_cube);
    }
}

/// Whether this device contracts `8×8×8` `f32` fragments and adds `f32` atomically into a
/// buffer, the two things the fragment fold needs; reported rather than silently passed.
fn sums_fragments(client: &cubecl::client::Client) -> bool {
    let f32_ty = f32::elem_type_native();
    let cmma = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == f32_ty
            && cfg.b_type == f32_ty
            && cfg.cd_type == f32_ty
            && cfg.m == 8
            && cfg.n == 8
            && cfg.k == 8
    });
    let adds = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add);
    if !(cmma && adds) {
        ValidationResult::Skipped(
            "device has no 8x8x8 f32 cmma fragment or no f32 atomic add".to_string(),
        )
        .enforce();
    }
    cmma && adds
}

/// `a·b` in fragments with `K` distributed to `splits` cubes, folded atomically into a zeroed
/// output.
fn run_atomic_split_cmma(k: usize, splits: usize) -> HostData {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let (m, n, edge) = (8usize, 8usize, 8usize);

    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 - 2.0).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    // One plane per cube, the whole output per cube, `K` distributed in runs of one stage.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, m), (N, n), (K, edge)])
                .walk_every(&[K])
                .cubes(&[M, N, K])
                .across(K, splits)
                .build(),
        ),
        Form::Static,
    );

    atomic_split_cmma::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

/// A fragment folds into the output through the scratch: every split sums to the whole
/// contraction, exactly, since the values are small integers.
#[test]
fn a_fragment_sums_into_the_output_through_the_scratch() {
    let client = cubecl::test_device().client();
    if !sums_fragments(&client) {
        return;
    }
    let (m, n, k) = (8usize, 8usize, 64usize);
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    for splits in [1usize, 2, 4, 8] {
        let got = run_atomic_split_cmma(k, splits);
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| a(i, p) * b(p, j)).sum();
                assert_eq!(got.get_f32(&[i, j]), want, "{splits} splits: ({i}, {j})");
            }
        }
    }
}

// -- A plain copy into a folding destination -------------------------------------------------
//
// `copy_from` is the memory-to-memory door, and it picks its path on the destination's shape. A
// folding destination has no address, so it must take the layout walk; before the write mode
// entered that condition a whole, unmasked, plain one took the straight path and panicked there.

/// Whether this device's buffers take an `f32` atomic add; reported rather than silently passed.
fn adds_atomically(client: &cubecl::client::Client) -> bool {
    let adds = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)))
        .contains(AtomicUsage::Add);
    if !adds {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
    }
    adds
}

/// Copy a whole source tile into a whole output whose writes add.
///
/// Whole on purpose: `at()` narrows the window and clears `Access::whole`, and it is the *whole*
/// destination that used to take the straight buffer fill. Narrowed first, this would take the
/// layout walk either way and prove nothing about the condition.
#[cube(launch)]
fn copy_into_summing<E: Numeric>(
    src: &TileArg<'_>,
    out: &AccumulateArg<'_, E>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let src = src.tile(&space);
    let mut out = out.tile(&space);
    out.copy_from(&src);
}

/// The whole claim: the copy adds, it does not replace, and it does not panic on the way. Drop
/// the write mode from the straight fill's condition and this panics about addresses instead.
#[test]
fn a_copy_into_a_summing_output_adds() {
    let client = cubecl::test_device().client();
    if !adds_atomically(&client) {
        return;
    }
    let dtype = f32::elem_type_native();
    let (rows, cols) = (4usize, 8usize);
    let values: Vec<f32> = (0..rows * cols).map(|i| (i % 5) as f32 + 1.0).collect();
    let (src, _) = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .custom(values.clone())
        .generate_with_f32_host_data();
    // Seeded, not zeroed: a store would erase the seed and an add keeps it, so the seed is
    // what tells the two apart.
    let seed = 10.0f32;
    let (out, _) = TestInput::builder(client.clone(), shape![rows, cols])
        .dtype(dtype)
        .custom(vec![seed; rows * cols])
        .generate_with_f32_host_data();

    // One cube over the whole tile, so the destination stays whole and each cell is added once.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, rows), (N, cols)]),
            Levels::leaf(&[(M, rows), (N, cols)]).cubes(&[M, N]).build(),
        ),
        Form::Static,
    );
    copy_into_summing::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            src.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]),
        ),
        AccumulateArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(
                got.get_f32(&[r, c]),
                values[r * cols + c] + seed,
                "at ({r}, {c}): the copy replaced the seed instead of adding to it"
            );
        }
    }
}
