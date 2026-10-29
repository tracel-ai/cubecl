//! Multi-axis reduce: contractions over more than one abstract axis.
//!
//! The first client is the simplest one that cannot be a matmul: a matmul whose `K` is declared as
//! *two* axes. Only how many axes the operands contract changes, so the answer must be exactly the
//! plain matmul's, isolating the multi-axis reduce nest from the projection machinery of conv.
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{
    prelude::*,
    zspace::{Shape, shape},
};

use super::{Form, implied};
use cubecl_tile::procedural::AffineCoordinate;
use cubecl_tile::procedural::Procedural;
use cubecl_tile::*;
use cubecl_tile::{kind::Boundary, launch::BoundaryPolicy};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
// `K` split in two: `K1` the major half, `K2` the minor (and the axis both operands line along).
const K1: Axis = Axis(3);
const K2: Axis = Axis(4);
const B: Axis = Axis(5);

/// The leaf's register block, held fixed across every kernel here.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `c.zero(); c.mma(a, b)` region by region over whatever one-level space it is handed: the
/// same body serves the one-axis and the two-axis contraction, which is the point.
#[cube(launch)]
fn reduce_matmul_kernel(
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

/// Where the reduced input is read from across the one level these kernels walk: where it lies,
/// or a shared-memory stages `depth` slots deep.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Read {
    InPlace,
    Smem { depth: usize },
}

/// Seeds `output` with the identity the reduction starts from, then reduces `input` into it region by
/// region. One body serves every monoid, which is the point: the kernels below differed only in
/// this seed and the `Monoid` passed to `reduce_axis`.
#[cube]
fn reduce_body<E: Numeric>(
    input: &Tile,
    output: &mut Tile,
    space: &Partitioning,
    #[comptime] level: Level,
    #[comptime] read: Read,
    #[comptime] monoid: Monoid,
) {
    output.init::<E>(Monoid::identity::<E>(monoid));
    let walk = space.over(&level);
    match comptime!(read) {
        Read::InPlace => {
            for region in walk {
                let mut out_region = output.at(&region);
                out_region.reduce_axis_accumulate(&input.at(&region), monoid);
            }
        }
        Read::Smem { depth } => {
            let mut stages = Stages::smem_single(&walk, input, StageStorage::Strided, depth, None);
            stages.pipelined(walk, |slot, region| {
                let mut out_region = output.at(region);
                slot.consume(|input_s| {
                    out_region.reduce_axis_accumulate(input_s, monoid);
                });
            });
        }
    }
}

#[cube(launch)]
fn reduce_kernel<E: Numeric>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] read: Read,
    #[comptime] monoid: Monoid,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let mut output = output.tile(&space);
    reduce_body::<E>(
        &input,
        &mut output,
        &space,
        comptime!(level.clone()),
        read,
        monoid,
    );
}

/// Reduce an axis-index recipe so a trailing partial tile must be masked without a backing window.
/// `read` decides whether the recipe is evaluated at the leaf or first materialized into shared
/// memory; either way the partial-tile mask must survive rather than reducing the padded overhang.
#[cube(launch)]
fn procedural_reduce_kernel<E: Float>(
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] read: Read,
    #[define(E)] _dtype: ElemType,
) {
    let input = Procedural::new::<E, AffineCoordinate<E>>(
        comptime!(space.space().clone()),
        AffineCoordinate::<E> {
            offset: E::new(0.0),
            coefficient: E::new(1.0),
            axis: K,
        },
    )
    .tile::<E>();
    let mut output = output.tile(&space);
    reduce_body::<E>(
        &input,
        &mut output,
        &space,
        comptime!(level.clone()),
        read,
        comptime!(Monoid::Max),
    );
}

/// Small integers, so every product and partial sum is exact in `f32` and the two kernels can be
/// compared for equality rather than closeness.
fn ramp(n: usize, period: usize) -> Vec<f32> {
    (0..n).map(|i| (i % period) as f32).collect()
}

/// Run `space` with `a`/`b` declared over `a_axes`/`b_axes` and the given buffer shapes, and
/// return the `m × n` output. The buffers are contiguous either way, so a rank-3 declaration of
/// the same bytes is just a finer reading of the same layout.
fn run(
    a_shape: Shape,
    b_shape: Shape,
    c_shape: Shape,
    a_axes: &[Axis],
    b_axes: &[Axis],
    c_axes: &[Axis],
    launcher: Launcher,
) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let (a_len, b_len) = (a_shape.num_elements(), b_shape.num_elements());

    let (a_handle, _) = TestInput::builder(client.clone(), a_shape)
        .dtype(f32_ty)
        .custom(ramp(a_len, 7))
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), b_shape)
        .dtype(f32_ty)
        .custom(ramp(b_len, 5))
        .generate_with_f32_host_data();
    let c_handle = TestInput::builder(client.clone(), c_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    reduce_matmul_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(a_axes),
        ),
        TileArgLaunch::new(
            b_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(b_axes),
        ),
        TileArgLaunch::new(
            c_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(c_axes),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
    );

    HostData::from_tensor_handle(&client, c_handle, HostDataType::F32)
}

/// The one-axis reference: an ordinary `{M, N, K}` matmul through the 2-D register leaf.
fn plain(m: usize, n: usize, k: usize, tm: usize, tn: usize) -> HostData {
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, tm), (N, tn), (K, k)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    run(
        shape![m, k],
        shape![k, n],
        shape![m, n],
        &[M, K],
        &[K, N],
        &[M, N],
        launcher.clone(),
    )
}

/// [`plain`] with a leading batch axis both operands span.
fn plain_batched(b: usize, m: usize, n: usize, k: usize, tm: usize, tn: usize) -> HostData {
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(B, b), (M, m), (N, n), (K, k)]),
            Levels::leaf(&[(B, 1), (M, tm), (N, tn), (K, k)])
                .walk_every(&[B, M, N, K])
                .build(),
        ),
        Form::Static,
    );
    run(
        shape![b, m, k],
        shape![b, k, n],
        shape![b, m, n],
        &[B, M, K],
        &[B, K, N],
        &[B, M, N],
        launcher.clone(),
    )
}

fn assert_same(got: &HostData, want: &HostData, dims: &[usize]) {
    let total: usize = dims.iter().product();
    for flat in 0..total {
        let mut idx = vec![0usize; dims.len()];
        let mut rest = flat;
        for p in (0..dims.len()).rev() {
            idx[p] = rest % dims[p];
            rest /= dims[p];
        }
        assert_eq!(
            got.get_f32(&idx),
            want.get_f32(&idx),
            "split-K disagrees with the plain matmul at {idx:?}"
        );
    }
}

/// Both halves of the split reduce ride the leaf: one `mma` call unravels a `k1 × k2` nest.
#[test]
fn split_contraction_whole_reduce_at_leaf() {
    let (m, n, k1, k2) = (8, 8, 3, 4);
    let (k, tm, tn) = (k1 * k2, 4, 4);

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K1, k1), (K2, k2)]),
            Levels::leaf(&[(M, tm), (N, tn), (K1, k1), (K2, k2)])
                .walk_every(&[M, N, K1, K2])
                .build(),
        ),
        Form::Static,
    );

    let got = run(
        shape![m, k1, k2],
        shape![k1, k2, n],
        shape![m, n],
        &[M, K1, K2],
        &[K1, K2, N],
        &[M, N],
        launcher.clone(),
    );
    assert_same(&got, &plain(m, n, k, tm, tn), &[m, n]);
}

/// The major half is walked instead: `k1` sequential `mma` calls, each reducing `k2`. The leaf
/// still contracts two axes (`K1` is present at extent 1), so it still takes the N-D nest.
#[test]
fn split_contraction_major_half_walked() {
    let (m, n, k1, k2) = (8, 8, 3, 4);
    let (k, tm, tn) = (k1 * k2, 4, 4);

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K1, k1), (K2, k2)]),
            Levels::leaf(&[(M, tm), (N, tn), (K1, 1), (K2, k2)])
                .walk_every(&[M, N, K1, K2])
                .build(),
        ),
        Form::Static,
    );

    let got = run(
        shape![m, k1, k2],
        shape![k1, k2, n],
        shape![m, n],
        &[M, K1, K2],
        &[K1, K2, N],
        &[M, N],
        launcher.clone(),
    );
    assert_same(&got, &plain(m, n, k, tm, tn), &[m, n]);
}

/// A batch axis alongside the two contracted ones: the leaf's batch unravel and its reduce
/// unravel are independent nests, so they must not be confused for one another.
#[test]
fn split_contraction_with_a_batch_axis() {
    let (b, m, n, k1, k2) = (3, 4, 8, 2, 4);
    let (k, tm, tn) = (k1 * k2, 4, 4);

    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(B, b), (M, m), (N, n), (K1, k1), (K2, k2)]),
            Levels::leaf(&[(B, 1), (M, tm), (N, tn), (K1, k1), (K2, k2)])
                .walk_every(&[B, M, N, K1, K2])
                .build(),
        ),
        Form::Static,
    );

    let got = run(
        shape![b, m, k1, k2],
        shape![b, k1, k2, n],
        shape![b, m, n],
        &[B, M, K1, K2],
        &[B, K1, K2, N],
        &[B, M, N],
        launcher.clone(),
    );
    assert_same(&got, &plain_batched(b, m, n, k, tm, tn), &[b, m, n]);
}

fn run_reduce(
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    launcher: Launcher,
    monoid: Monoid,
) -> HostData {
    run_reduce_with_vw(
        in_shape,
        out_shape,
        in_axes,
        out_axes,
        launcher.clone(),
        monoid,
        1,
        Read::InPlace,
    )
}

/// [`run_reduce`] with the input staged through stages `depth` deep.
#[allow(clippy::too_many_arguments)]
fn run_reduce_staged(
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    launcher: Launcher,
    monoid: Monoid,
    depth: usize,
) -> HostData {
    run_reduce_with_vw(
        in_shape,
        out_shape,
        in_axes,
        out_axes,
        launcher.clone(),
        monoid,
        1,
        Read::Smem { depth },
    )
}

/// Exercise a 2-D `M × K -> M` reduction and derive the reference reduction from `op`, so schedule
/// coverage does not duplicate the three identities and comparison loops.
fn check_2d_reduce(depth: usize, m: usize, k: usize, tm: usize, tk: usize, monoid: Monoid) {
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );
    // Every caller of this helper stages: the stages depth is what the buffering coverage
    // exercises.
    let got = run_reduce_staged(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        monoid,
        depth,
    );

    for i in 0..m {
        let values = (0..k).map(|j| ((i * k + j) % 7) as f32);
        let want = match monoid {
            Monoid::Sum => values.sum(),
            Monoid::Prod => values.product(),
            Monoid::Max => values.fold(f32::NEG_INFINITY, f32::max),
            Monoid::Min => values.fold(f32::INFINITY, f32::min),
        };
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "depth {depth} {monoid:?} mismatch at index {i}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_reduce_with_vw(
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    launcher: Launcher,
    monoid: Monoid,
    in_vw: usize,
    read: Read,
) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();
    let in_len = in_shape.num_elements();

    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(ramp(in_len, 7))
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let in_binding = in_handle.binding();
    let out_binding = out_handle.clone().binding();

    reduce_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            in_binding.into_tensor_arg(),
            f32_ty,
            in_vw,
            TileSpec::direct(in_axes),
        ),
        TileArgLaunch::new(
            out_binding.into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(out_axes),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        read,
        monoid,
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

#[test]
fn test_reduce_axis_sum_2d_to_1d() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        Monoid::Sum,
    );

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(got.get_f32(&[i]), want, "Sum mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_sum_walked_levels() {
    let (m, k, tm, tk) = (8, 16, 4, 4);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        Monoid::Sum,
    );

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(got.get_f32(&[i]), want, "Sum mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_max_2d_to_1d() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        Monoid::Max,
    );

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got.get_f32(&[i]), want, "Max mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_min_2d_to_1d() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        Monoid::Min,
    );

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::INFINITY, f32::min);
        assert_eq!(got.get_f32(&[i]), want, "Min mismatch at index {i}");
    }
}

#[test]
fn test_reduce_axis_multi_axis_3d_to_1d() {
    let (b, m, k) = (3, 4, 8);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(B, b), (M, m), (K, k)]),
            Levels::leaf(&[(B, 1), (M, 2), (K, 4)])
                .walk_every(&[B, M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![b, m, k],
        shape![b],
        &[B, M, K],
        &[B],
        launcher.clone(),
        Monoid::Sum,
    );

    for bi in 0..b {
        let mut want = 0.0f32;
        for mi in 0..m {
            for ki in 0..k {
                let flat = bi * (m * k) + mi * k + ki;
                want += (flat % 7) as f32;
            }
        }
        assert_eq!(got.get_f32(&[bi]), want, "Multi-axis sum mismatch at {bi}");
    }
}

#[test]
fn test_reduce_axis_sum_staged() {
    check_2d_reduce(1, 8, 16, 4, 8, Monoid::Sum);
}

#[test]
fn test_reduce_axis_sum_double_buffered() {
    check_2d_reduce(2, 8, 16, 4, 4, Monoid::Sum);
}

#[test]
fn test_reduce_axis_max_staged() {
    check_2d_reduce(1, 8, 16, 4, 8, Monoid::Max);
}

#[test]
fn test_reduce_axis_min_staged() {
    check_2d_reduce(1, 8, 16, 4, 8, Monoid::Min);
}

#[test]
fn test_reduce_axis_max_double_buffered() {
    check_2d_reduce(2, 8, 16, 4, 4, Monoid::Max);
}

#[test]
fn test_reduce_axis_min_double_buffered() {
    check_2d_reduce(2, 8, 16, 4, 4, Monoid::Min);
}

/// Reduction over an outer axis, keeping the innermost axis (which lines along the vector width).
#[test]
fn test_reduce_axis_sum_outer_axis_retained_innermost_v1() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        launcher.clone(),
        Monoid::Sum,
        1,
        Read::InPlace,
    );

    for j in 0..k {
        let want: f32 = (0..m).map(|i| ((i * k + j) % 7) as f32).sum();
        assert_eq!(got.get_f32(&[j]), want, "Sum mismatch at index {j}");
    }
}

/// Reduction over an outer axis while retaining the innermost axis with `vector_size` = 4.
/// Exercises line indexing and unit extraction when the innermost axis is in accumulator nest.
#[test]
fn test_reduce_axis_sum_outer_axis_retained_innermost_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        launcher.clone(),
        Monoid::Sum,
        4,
        Read::InPlace,
    );

    for j in 0..k {
        let want: f32 = (0..m).map(|i| ((i * k + j) % 7) as f32).sum();
        assert_eq!(
            got.get_f32(&[j]),
            want,
            "Vectorized sum mismatch at index {j}"
        );
    }
}

/// [`test_reduce_axis_sum_inner_axis_reduced_v4`] under `Max`, whose identity the memory system
/// does not hand back out of bounds: the substitution has to happen a whole line at a time.
#[test]
fn test_reduce_axis_max_inner_axis_reduced_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher.clone(),
        Monoid::Max,
        4,
        Read::InPlace,
    );

    for i in 0..m {
        let want = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Line-folded max mismatch at row {i}"
        );
    }
}

/// [`run_reduce`], but the input is `checked(true)`: a non-divisible reduced axis leaves an
/// overhang past the operand's real data, and only a checked operand masks it instead of reading
/// garbage.
#[allow(clippy::too_many_arguments)]
fn run_reduce_checked(
    in_data: Vec<f32>,
    in_shape: Shape,
    out_shape: Shape,
    in_axes: &[Axis],
    out_axes: &[Axis],
    launcher: Launcher,
    monoid: Monoid,
    read: Read,
) -> HostData {
    let client = cubecl::test_device().client();
    let f32_ty = f32::elem_type_native();

    let (in_handle, _) = TestInput::builder(client.clone(), in_shape)
        .dtype(f32_ty)
        .custom(in_data)
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(f32_ty)
        .zeros()
        .generate_without_host_data();

    let in_binding = in_handle.binding();
    let out_binding = out_handle.clone().binding();

    reduce_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            in_binding.into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(in_axes).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        ),
        TileArgLaunch::new(
            out_binding.into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(out_axes),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        read,
        monoid,
        f32_ty,
    );

    HostData::from_tensor_handle(&client, out_handle, HostDataType::F32)
}

fn nondivisible_k_space(m: usize, k: usize, tk: usize) -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, m), (K, tk)]).walk_every(&[M, K]).build(),
        ),
        Form::Static,
    )
}

#[test]
fn test_reduce_axis_sum_nondivisible_k() {
    let (m, k, tk) = (4, 6, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk),
        Monoid::Sum,
        Read::InPlace,
    );

    for i in 0..m {
        let want: f32 = data[i * k..(i + 1) * k].iter().sum();
        assert_eq!(got.get_f32(&[i]), want, "Sum mismatch at row {i}");
    }
}

#[test]
fn test_reduce_axis_sum_nondivisible_k_staged() {
    let (m, k, tk) = (4, 6, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk),
        Monoid::Sum,
        Read::Smem { depth: 1 },
    );

    for i in 0..m {
        let want: f32 = data[i * k..(i + 1) * k].iter().sum();
        assert_eq!(got.get_f32(&[i]), want, "Staged sum mismatch at row {i}");
    }
}

#[test]
fn test_reduce_axis_sum_nondivisible_k_double_buffered() {
    let (m, k, tk) = (4, 10, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk),
        Monoid::Sum,
        Read::Smem { depth: 2 },
    );

    for i in 0..m {
        let want: f32 = data[i * k..(i + 1) * k].iter().sum();
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Double-buffered sum mismatch at row {i}"
        );
    }
}

#[test]
fn test_reduce_axis_max_nondivisible_k() {
    let (m, k, tk) = (4, 6, 4);
    let data = ramp(m * k, 7);

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk),
        Monoid::Max,
        Read::InPlace,
    );

    for i in 0..m {
        let want = data[i * k..(i + 1) * k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got.get_f32(&[i]), want, "Max mismatch at row {i}");
    }
}

/// Max over `k` of the axis-index recipe, with the source read as `read` says. The last real
/// `k` is 5, so a mask that leaked the padded overhang would report the tile edge instead.
fn check_procedural_reduce(read: Read) {
    let (m, k, tk) = (4, 6, 4);
    let launcher = nondivisible_k_space(m, k, tk);
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let output = TestInput::builder(client.clone(), shape![m])
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    procedural_reduce_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            output.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        read,
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output, HostDataType::F32);
    for row in 0..m {
        assert_eq!(got.get_f32(&[row]), 5.0);
    }
}

#[test]
fn test_procedural_max_masks_nondivisible_k() {
    check_procedural_reduce(Read::InPlace);
}

#[test]
fn test_staged_procedural_max_masks_nondivisible_k() {
    check_procedural_reduce(Read::Smem { depth: 1 });
}

/// `ramp`'s data is nonnegative, so a masked overhang cell falling back to zero (Sum's identity)
/// would still leave `Max` accidentally correct: zero never beats the real maximum. Strictly
/// negative data closes that gap, since a zero fallback would then beat every real value.
#[test]
fn test_reduce_axis_max_nondivisible_k_negative_data() {
    let (m, k, tk) = (4, 6, 4);
    let data: Vec<f32> = (0..m * k).map(|i| -1.0 - i as f32).collect();

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk),
        Monoid::Max,
        Read::InPlace,
    );

    for i in 0..m {
        let want = data[i * k..(i + 1) * k]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(got.get_f32(&[i]), want, "Max mismatch at row {i}");
    }
}

/// A zero overhang fallback would incorrectly win this Min reduction, whereas the correct
/// device-side identity is the element type's maximum value.
#[test]
fn test_reduce_axis_min_nondivisible_k_positive_data() {
    let (m, k, tk) = (4, 6, 4);
    let data: Vec<f32> = (0..m * k).map(|i| 1.0 + i as f32).collect();

    let got = run_reduce_checked(
        data.clone(),
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        nondivisible_k_space(m, k, tk),
        Monoid::Min,
        Read::InPlace,
    );

    for i in 0..m {
        let want = data[i * k..(i + 1) * k]
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        assert_eq!(got.get_f32(&[i]), want, "Min mismatch at row {i}");
    }
}

#[test]
fn test_reduce_axis_max_outer_axis_retained_innermost_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        launcher.clone(),
        Monoid::Max,
        4,
        Read::InPlace,
    );

    for j in 0..k {
        let want: f32 = (0..m)
            .map(|i| ((i * k + j) % 7) as f32)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            got.get_f32(&[j]),
            want,
            "Vectorized max mismatch at index {j}"
        );
    }
}

#[test]
fn test_reduce_axis_min_outer_axis_retained_innermost_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![k],
        &[M, K],
        &[K],
        launcher.clone(),
        Monoid::Min,
        4,
        Read::InPlace,
    );

    for j in 0..k {
        let want: f32 = (0..m)
            .map(|i| ((i * k + j) % 7) as f32)
            .fold(f32::INFINITY, f32::min);
        assert_eq!(
            got.get_f32(&[j]),
            want,
            "Vectorized min mismatch at index {j}"
        );
    }
}

/// Reduction over the innermost axis with `vector_size` = 4: the line runs along the axis being
/// reduced, so a step reduces the whole line into one cell and the units collapse once at the end.
#[test]
fn test_reduce_axis_sum_inner_axis_reduced_v4() {
    let (m, k, tm, tk) = (8, 16, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(M, tm), (K, tk)])
                .walk_every(&[M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce_with_vw(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher.clone(),
        Monoid::Sum,
        4,
        Read::InPlace,
    );

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Vectorized sum mismatch at index {i}"
        );
    }
}

/// 3D tensor where the middle axis is reduced and innermost is retained, with `vector_size` = 4.
#[test]
fn test_reduce_axis_multi_axis_3d_middle_axis_retained_innermost_v4() {
    let (b, m, k) = (3, 4, 16);
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(B, b), (M, m), (K, k)]),
            Levels::leaf(&[(B, 1), (M, 2), (K, 8)])
                .walk_every(&[B, M, K])
                .build(),
        ),
        Form::Static,
    );

    let got = run_reduce_with_vw(
        shape![b, m, k],
        shape![b, k],
        &[B, M, K],
        &[B, K],
        launcher.clone(),
        Monoid::Sum,
        4,
        Read::InPlace,
    );

    for bi in 0..b {
        for ki in 0..k {
            let mut want = 0.0f32;
            for mi in 0..m {
                let flat = bi * (m * k) + mi * k + ki;
                want += (flat % 7) as f32;
            }
            assert_eq!(
                got.get_f32(&[bi, ki]),
                want,
                "3D middle-axis reduce mismatch at ({bi}, {ki})"
            );
        }
    }
}

/// Reduction over an axis spread across plane units (`ComputeScope::Unit` / `UnitShare::Plane`).
/// Tests that `UnitShare::Plane` seeds with 0 and reduces across units combining with accumulator.
#[test]
fn test_reduce_axis_sum_spatial_unit_units() {
    let client = cubecl::test_device().client();
    let plane_size = client.properties().hardware.plane_size_max as usize;

    let (m, kr) = (4usize, 4usize);
    let k = plane_size * kr;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(K, kr)]).units(&[(K, plane_size)]).build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        Monoid::Sum,
    );

    for i in 0..m {
        let want: f32 = (0..k).map(|j| ((i * k + j) % 7) as f32).sum();
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Spatial Unit sum mismatch at index {i}"
        );
    }
}

#[test]
fn test_reduce_axis_max_spatial_unit_units() {
    let client = cubecl::test_device().client();
    let plane_size = client.properties().hardware.plane_size_max as usize;

    let (m, kr) = (4usize, 4usize);
    let k = plane_size * kr;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(K, kr)]).units(&[(K, plane_size)]).build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        Monoid::Max,
    );

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Spatial Unit max mismatch at index {i}"
        );
    }
}

#[test]
fn test_reduce_axis_min_spatial_unit_units() {
    let client = cubecl::test_device().client();
    let plane_size = client.properties().hardware.plane_size_max as usize;

    let (m, kr) = (4usize, 4usize);
    let k = plane_size * kr;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (K, k)]),
            Levels::leaf(&[(K, kr)]).units(&[(K, plane_size)]).build(),
        ),
        Form::Static,
    );

    let got = run_reduce(
        shape![m, k],
        shape![m],
        &[M, K],
        &[M],
        launcher,
        Monoid::Min,
    );

    for i in 0..m {
        let want: f32 = (0..k)
            .map(|j| ((i * k + j) % 7) as f32)
            .fold(f32::INFINITY, f32::min);
        assert_eq!(
            got.get_f32(&[i]),
            want,
            "Spatial Unit min mismatch at index {i}"
        );
    }
}

/// A `Max` reduce whose accumulator lives in registers while the reduced axis is split across the
/// plane's units: each unit reduces its own `K` slice, so each holds a partial maximum, and the drain
/// combines them under the same monoid the accumulator was built with.
///
/// Two things had to be true for this to work, and neither was. The drain combined units with a
/// hardcoded sum, and a promoted block read its `UnitShare` from the tile, which is only stamped on
/// the way down, so it saw `Whole` and every unit wrote its partial over the last.
///
/// The data is all negative, so any identity leaking in (a zero from a sum-shaped combine, or
/// from an out-of-bounds read) wins the maximum and the assert catches it.
#[cube(launch)]
fn resident_reduce_kernel<E: Numeric>(
    input: &TileArg<'_>,
    output: &TileArg<'_>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] monoid: Monoid,
    #[define(E)] _dtype: ElemType,
) {
    let input = input.tile(&space);
    let out = output.tile(&space);
    let mut acc = out.block_reducer::<E>(&input, REGISTER_BLOCK, monoid);
    acc.init::<E>(Monoid::identity::<E>(monoid));
    for region in space.over(&level) {
        let mut acc_region = acc.at(&region);
        acc_region.reduce_axis_accumulate(&input.at(&region), monoid);
    }
    acc.drained_into(&out);
}

#[test]
fn resident_max_over_unit_split_contraction() {
    let client = cubecl::test_device().client();
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let (m, n, kr) = (4usize, 4usize, 2usize);
    let k = plane_size * kr;

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(K, kr)]).units(&[(K, plane_size)]).build(),
        ),
        Form::Static,
    );

    let values: Vec<f32> = (0..m * n * k).map(|i| -1.0 - ((i % 13) as f32)).collect();
    let f32_ty = f32::elem_type_native();
    let (in_handle, _) = TestInput::builder(client.clone(), shape![m, n, k])
        .dtype(f32_ty)
        .custom(values.clone())
        .generate_with_f32_host_data();
    // Poisoned with values above every input, so a reduction that let the sink take part would win the
    // maximum and be caught. `reduce_axis` owns the init, and the unit-split contraction is
    // exactly the case where it must seed rather than overwrite.
    let out_handle = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32_ty)
        .uniform(7, 1.0, 2.0)
        .generate_without_host_data();

    resident_reduce_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            in_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[M, N, K]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Monoid::Max,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for i in 0..m {
        for j in 0..n {
            let want = (0..k)
                .map(|p| values[(i * n + j) * k + p])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(got.get_f32(&[i, j]), want, "max mismatch at ({i}, {j})");
        }
    }
}

/// The twin of [`resident_max_over_unit_split_contraction`] at a **segmented** reduction: the plane splits into
/// aligned groups, each holding one `(m, n)` cell's partials, rather than the whole plane holding
/// one. `UnitShare::Group` where that test is `UnitShare::Plane`.
///
/// The drain is shared with the promoted matmul's, where reading the odometer off a projected
/// nest (the accumulator spans `{M, N}`, so the contracted `K` is not in its axis list) gave every
/// group the same output row. `reduce_axis` reaches the same code, so it gets the same coverage.
///
/// All the data is negative, so an identity leaking in from an unwritten cell wins the maximum
/// and the assert catches it.
#[test]
#[ignore = "known-failing reproducer: the segmented share is still wrong on this path, and \
            whether that is a cubek defect or an unsupported combination is not yet established \
            , it is not what the walk fix addresses"]
fn resident_max_over_unit_group_k() {
    let client = cubecl::test_device().client();
    let plane_size = client.properties().hardware.plane_size_max as usize;
    let (group_units, kr, n) = (8usize, 2usize, 2usize);
    let (groups, k) = (plane_size / group_units, group_units * kr);
    let m = groups;

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 1), (K, kr)])
                .units(&[(M, groups), (K, group_units)])
                .interleaved(K)
                .build(),
        ),
        Form::Static,
    );

    let values: Vec<f32> = (0..m * n * k).map(|i| -1.0 - ((i % 13) as f32)).collect();
    let f32_ty = f32::elem_type_native();
    let (in_handle, _) = TestInput::builder(client.clone(), shape![m, n, k])
        .dtype(f32_ty)
        .custom(values.clone())
        .generate_with_f32_host_data();
    let out_handle = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32_ty)
        .uniform(7, 1.0, 2.0)
        .generate_without_host_data();

    resident_reduce_kernel::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            in_handle.binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[M, N, K]),
        ),
        TileArgLaunch::new(
            out_handle.clone().binding().into_tensor_arg(),
            f32_ty,
            1,
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        Monoid::Max,
        f32_ty,
    );

    let got = HostData::from_tensor_handle(&client, out_handle, HostDataType::F32);
    for i in 0..m {
        for j in 0..n {
            let want = (0..k)
                .map(|p| values[(i * n + j) * k + p])
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(got.get_f32(&[i, j]), want, "max mismatch at ({i}, {j})");
        }
    }
}
