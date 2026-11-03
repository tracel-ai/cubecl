//! Unit tests for [`Space`]

use super::{Form, implied};
use cubecl::prelude::*;
use cubecl_tile::{Axis, Launcher, Level, Levels, Partitioning, Space};

// Matmul-style axis labels reused across the cases below. `B0`/`B1` are two
// independent batch axes (a batch is just ordinary axes; broadcasting is omission).
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B0: Axis = Axis(3);
const B1: Axis = Axis(4);
// A second contracted axis (a partitioned `K`), and one a walk routes.
const K2: Axis = Axis(5);
const EXPERT: Axis = Axis(6);

// ---- Space ----------------------------------------------------------------

#[test]
fn new_builds_plain_axes() {
    let space = Space::new(&[(M, 4), (N, 8), (K, 2)]);
    assert_eq!(space.rank(), 3);
    assert_eq!(space.extent(M), 4);
    assert_eq!(space.extent(N), 8);
    assert_eq!(space.extent(K), 2);
}

#[test]
fn project_keeps_listed_axes_in_order() {
    let space = Space::new(&[(B0, 12), (M, 16), (K, 8)]);
    let lhs = space.subspace(&[B0, M, K]);
    assert_eq!(lhs.rank(), 3);
    assert_eq!(lhs.extent(B0), 12);
    assert_eq!(lhs.extent(M), 16);

    // An operand broadcasts a batch axis by simply leaving it out of the projection.
    let dropped = space.subspace(&[M, K]);
    assert_eq!(dropped.rank(), 2);
    assert!(!dropped.contains(B0));
}

#[test]
fn merge_assembles_two_batch_broadcast() {
    // lhs carries B0, rhs carries B1; each omits (broadcasts) the other's batch axis.
    // The merge rebuilds the full {B0, B1, M, N, K} operation space.
    let lhs = Space::new(&[(B0, 4), (M, 16), (K, 8)]);
    let rhs = Space::new(&[(B1, 3), (K, 8), (N, 4)]);
    let out = Space::new(&[(B0, 4), (B1, 3), (M, 16), (N, 4)]);

    let merged = Space::merge(&[&out, &lhs, &rhs]);
    assert_eq!(merged.rank(), 5);
    assert_eq!(merged.extent(B0), 4);
    assert_eq!(merged.extent(B1), 3);
    assert_eq!(merged.extent(K), 8);
    // First-appearance order: out's axes lead, K (only on the operands) comes last.
    assert_eq!(merged.axis_at(0), B0);
    assert_eq!(merged.axis_at(1), B1);
    assert_eq!(merged.axis_at(4), K);
}

#[test]
fn merge_shared_size_one_axis_broadcasts() {
    // A shared axis where one side is size 1 broadcasts to the other (numpy rule).
    let lhs = Space::new(&[(B0, 1), (M, 16)]);
    let rhs = Space::new(&[(B0, 4), (M, 16)]);
    assert_eq!(Space::merge(&[&lhs, &rhs]).extent(B0), 4);
}

#[test]
fn merge_omitted_axis_broadcasts_wholesale() {
    let lhs = Space::new(&[(B0, 12), (M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 8), (N, 4)]);

    let merged = Space::merge(&[&lhs, &rhs]);
    assert!(merged.contains(B0));
    assert_eq!(merged.extent(B0), 12);
}

#[test]
fn merge_plain_shared_axis_agrees() {
    let lhs = Space::new(&[(M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 8), (N, 4)]);
    let merged = Space::merge(&[&lhs, &rhs]);
    assert_eq!(merged.extent(K), 8);
    assert_eq!(merged.rank(), 3);
}

#[test]
#[should_panic(expected = "conflicting extents")]
fn merge_conflicting_extent_panics() {
    let lhs = Space::new(&[(M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 4), (N, 4)]); // K disagrees: 8 vs 4
    let _ = Space::merge(&[&lhs, &rhs]);
}

// ---- Level::child (the tiling scheme) --------------------------------------

#[test]
fn a_level_cuts_each_axis_to_its_tile() {
    let level = Level::every(&[(M, 4), (N, 3), (K, 2)]);
    let tile = level.child(&Space::new(&[(M, 16), (N, 12), (K, 8)]));
    assert_eq!(tile.extent(M), 4);
    assert_eq!(tile.extent(N), 3);
    assert_eq!(tile.extent(K), 2);
    assert_eq!(tile.rank(), 3);
}

#[test]
fn levels_chain_into_a_multi_level_scheme() {
    let space = Space::new(&[(M, 64), (N, 64)]);
    let levels = Levels::leaf(&[(M, 4), (N, 4)])
        .walk(&[(M, 4), (N, 4)])
        .walk_every(&[M, N])
        .build();
    let level1 = levels[0].child(&space);
    let level2 = levels[1].child(&level1);

    assert_eq!(level1.extent(M), 16);
    assert_eq!(level2.extent(M), 4);
    assert_eq!(level2.extent(N), 4);
    assert_eq!(Partitioning::new(space.clone(), levels).leaf(), level2);
}

// ---- contracting ----------------------------------------------------------

/// A routed axis, as the leaf sees it: it holds one value, only the rhs spans it, and the output
/// does not carry it. It pairs with nothing and contracts nothing, so it is not one of the
/// contracted axes, and the fastest one is the one both operands line along.
#[test]
fn a_routed_axis_is_not_contracted() {
    let lhs = Space::new(&[(M, 1), (K, 8)]);
    let rhs = Space::new(&[(EXPERT, 1), (N, 4), (K, 8)]);
    let out = Space::new(&[(M, 1), (N, 4)]);
    assert_eq!(&Space::contracted(&[&lhs, &rhs], &out)[..], &[K]);
}

/// An rhs-only axis holding more than one value is a contraction the walk has to step, and
/// stays.
#[test]
fn an_axis_one_operand_spans_is_contracted_while_it_varies() {
    let lhs = Space::new(&[(M, 1), (K, 8)]);
    let rhs = Space::new(&[(EXPERT, 3), (N, 4), (K, 8)]);
    let out = Space::new(&[(M, 1), (N, 4)]);
    assert_eq!(&Space::contracted(&[&lhs, &rhs], &out)[..], &[K, EXPERT]);
}

/// A one-tap window on both axes of a separable filter, as nearest interpolation contracts: every
/// operand spans both axes, so both stay, and each factor keeps the axis it names by position.
#[test]
fn a_one_value_axis_every_operand_spans_is_contracted() {
    let weights = Space::new(&[(M, 4), (K, 1), (K2, 1)]);
    let input = Space::new(&[(M, 4), (K, 1), (K2, 1), (N, 4)]);
    let out = Space::new(&[(M, 4), (N, 4)]);
    assert_eq!(&Space::contracted(&[&weights, &input], &out)[..], &[K, K2]);
}

// ---- overhangs -------------------------------------------------------------

/// A CPU-matmul-shaped scheme: cube tiles of `planes x leaf` leaves over `(m, n, k)`, `K` handed
/// down whole to the cube and stepped by `leaf_k` under the plane grid.
fn cpu_matmul_nest(m: usize, n: usize, k: usize) -> Launcher {
    let (leaf_m, leaf_n, leaf_k) = (8, 8, 4);
    let (planes_m, planes_n) = (2, 4);
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, leaf_m), (N, leaf_n), (K, leaf_k)])
                .walk_every(&[K])
                .walk(&[(M, planes_m), (N, planes_n)])
                .walk_every(&[M, N])
                .build(),
        ),
        Form::Static,
    )
}

fn hangs(launcher: &Launcher, axis: Axis) -> bool {
    launcher.partitioning().overhangs(axis)
}

#[test]
fn overhangs_matches_cpu_matmul_checks() {
    // Every level divides: cube tiles 16×32, leaves 8×8×4.
    let launcher = cpu_matmul_nest(64, 64, 16);
    assert!(!hangs(&launcher, M));
    assert!(!hangs(&launcher, N));
    assert!(!hangs(&launcher, K));

    // m = 40 is not a multiple of the cube tile (16): M overhangs (the CPU matmul's check_m).
    // Within a cube the plane split is exact by construction, so the leaf level adds nothing.
    assert!(hangs(&cpu_matmul_nest(40, 64, 16), M));

    // K is handed to the cube whole (nothing to overhang); k = 18 fails only at the leaf
    // (leaf_k = 4): the deeper every-level alone drives the overhang (the CPU matmul's check_k).
    let launcher = cpu_matmul_nest(64, 64, 18);
    assert!(hangs(&launcher, K));
    assert!(!hangs(&launcher, M));
}

#[test]
fn overhangs_with_no_level_never() {
    // No level: nothing to misdivide.
    assert!(!Partitioning::new(Space::new(&[(M, 7)]), vec![]).overhangs(M));
}

#[test]
#[should_panic(expected = "concrete space")]
fn overhangs_dynamic_axis_panics() {
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, 64)]).all_dynamic(),
            Levels::leaf(&[(M, 16)]).walk_every(&[M]).build(),
        ),
        Form::Static,
    );
    let _ = hangs(&launcher, M);
}

// ---- Level constructors ----------------------------------------------------------

/// The tiles of several axes distributed as one index: the shares ride the cubes even though no
/// axis does, so the launch grid is their count.
#[test]
fn shared_tiles_launch_their_instances() {
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 16)]),
            Levels::leaf(&[(M, 16), (N, 32), (K, 4)])
                .walk(&[(K, 4)])
                .cubes(&[M, N, K])
                .shared_by(5)
                .build(),
        ),
        Form::Static,
    );
    assert!(launcher.partitioning().level(0).shared_by().is_some());
    // Five cubes, not `4 * 2 * 1`.
    assert!(matches!(launcher.cube_count(), CubeCount::Static(5, 1, 1)));
    assert_eq!(launcher.partitioning().levels().len(), 2);
}

/// Batch axes ride `Z` one tile each, however many there are and however they are listed: a
/// box of the grid, not a share.
#[test]
fn batches_are_a_dial_each() {
    let one_line = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(B0, 2), (B1, 3), (M, 64), (N, 64), (K, 16)]),
            Levels::leaf(&[(M, 16), (N, 32)])
                .cubes(&[M, N])
                .batches(&[B0, B1])
                .build(),
        ),
        Form::Static,
    );
    let a_dial_each = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(B0, 2), (B1, 3), (M, 64), (N, 64), (K, 16)]),
            Levels::leaf(&[(M, 16), (N, 32)])
                .cubes(&[M, N])
                .batches(&[B0])
                .batches(&[B1])
                .build(),
        ),
        Form::Static,
    );

    assert_eq!(
        one_line.partitioning().levels(),
        a_dial_each.partitioning().levels()
    );
    // No work: the lowering that reads this is the one that picks the per-region accumulator
    // nest.
    assert!(one_line.partitioning().level(0).shared_by().is_none());
    // Both axes ride Z, one cube per (B0, B1) pair, behind the `4 x 2` grid on X and Y.
    assert!(matches!(one_line.cube_count(), CubeCount::Static(4, 2, 6)));
}

/// One axis is a box whatever the count, so `across` on it distributes the axis's own tiles over
/// the scope, which is what a cut has always meant: no work is stated.
#[test]
fn one_axis_across_a_count_is_a_dial() {
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 16)]),
            Levels::leaf(&[(M, 16), (N, 32)])
                .walk_every(&[N])
                .cubes(&[M])
                .across(M, 4)
                .build(),
        ),
        Form::Static,
    );
    assert!(launcher.partitioning().level(0).shared_by().is_none());
    assert!(matches!(launcher.cube_count(), CubeCount::Static(4, 1, 1)));
}

/// Nothing named is nothing said: a level that names no axis cuts every cube the whole space.
#[test]
fn a_level_naming_no_axis_distributes_everything_to_one_cube() {
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64), (K, 16)]),
            Levels::leaf(&[(M, 16), (N, 32), (K, 16)])
                .walk_every(&[M, N, K])
                .cubes(&[])
                .build(),
        ),
        Form::Static,
    );
    assert!(launcher.partitioning().level(0).shared_by().is_none());
    assert!(matches!(launcher.cube_count(), CubeCount::Static(1, 1, 1)));
    assert_eq!(
        &launcher.partitioning().level(0).child(launcher.space()),
        launcher.space()
    );
}

/// The plane's units combine in registers, which needs them in lockstep. Units holding different
/// shares are on different regions, so they never reach a reduction together.
#[test]
#[should_panic = "combine in registers"]
fn sharing_tiles_across_units_is_refused() {
    let _ = Levels::leaf(&[(M, 16), (N, 32), (K, 16)])
        .units(&[(M, 4), (N, 4), (K, 4)])
        .shared_by(4)
        .build();
}

/// A share is a run of one index, so its entries say nothing of their own: a count or a spread on
/// one of them would size a box a share has no use for.
#[test]
#[should_panic = "states a count or a spread of its own"]
fn sharing_tiles_with_a_knob_on_an_entry_is_refused() {
    let _ = Levels::leaf(&[(M, 16), (N, 32), (K, 16)])
        .cubes(&[M, N, K])
        .interleaved(M)
        .shared_by(5)
        .build();
}

/// A level states each of its axes once, whichever way it states them.
#[test]
#[should_panic = "a level states each of its axes once"]
fn an_axis_named_twice_is_refused() {
    let _ = Levels::leaf(&[(M, 16)]).cubes(&[M, M]).build();
}

// ---- A level that cuts nothing --------------------------------------------

/// A level that takes one of the thing below has one region on every axis, so it partitions
/// nothing. It stays all the same: the kernel walks the levels it stated, one loop per level, so
/// the list must hold every one. A one-region walk evaluates away in the kernel, so it costs nothing.
#[test]
fn a_level_that_cuts_nothing_is_kept() {
    let plain = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64)]),
            Levels::leaf(&[(M, 16), (N, 32)])
                .walk_every(&[M, N])
                .build(),
        ),
        Form::Static,
    );
    let launcher = implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, 64), (N, 64)]),
            Levels::leaf(&[(M, 16), (N, 32)])
                // One of the tile below: nothing left to cut, still a level.
                .walk(&[(M, 1), (N, 1)])
                .walk_every(&[M, N])
                .build(),
        ),
        Form::Static,
    );

    assert_ne!(
        launcher.partitioning().levels(),
        plain.partitioning().levels()
    );
    assert_eq!(launcher.partitioning().levels().len(), 2);
    assert_eq!(
        launcher
            .partitioning()
            .level(0)
            .child(launcher.space())
            .extent(M),
        16
    );
    assert_eq!(launcher.partitioning().leaf().extent(M), 16);
}
