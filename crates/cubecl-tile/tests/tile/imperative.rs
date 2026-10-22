//! The walk written by the kernel: the levels are loops, the stages are stages the kernel
//! allocates, and the leaf runs under the register block the kernel states. The reference every
//! routine's rewrite is measured against.
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput, TileInput, assert_equals_approx};
use cubecl::{ir::OpaqueType, prelude::*, zspace::shape};
use cubecl_tile::launch::{Grid, Refusal};
use cubecl_tile::stage::Role;
use cubecl_tile::*;

use super::references;
use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// The software instruction the leaf runs: a 16-cell budget, no edge split, no unit fan-out.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// `c = a · b` over a K walk whose blocks of `a` and `b` are double-buffered in shared memory,
/// the leaf running the software instruction on the block's final tiles.
#[cube(launch)]
fn ring_matmul(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[comptime] depth: usize,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    c.zero();

    // The cube's walk: one block of K per region, both operands staged for it.
    let walk = space.walk();
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, depth, None);
    stages.pipelined(walk, |slot, region| {
        let c_block = c.at(region);
        slot.consume(|a_s, b_s| {
            // The block's own grid of final tiles, each contracted by the leaf.
            for cell in region {
                let mut c_cell = c_block
                    .at(&cell)
                    .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
                c_cell.mma(&a_s.at(&cell), &b_s.at(&cell));
            }
        });
    });
}

/// The same contraction with the stages' two steps written out and the plane's role read off it:
/// the shape a specialized kernel takes. With no plane set aside to fill, every plane computes,
/// so the fill arm is emitted and never entered and the compute arm does both halves itself.
#[cube(launch)]
fn role_split_matmul(a: &TileArg<'_>, b: &TileArg<'_>, c: &TileArg<'_>, space: Partitioning) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let mut c = c.tile(&space);
    c.zero();

    let walk = space.walk();
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize, None);

    match stages.role() {
        Role::Fill => {
            for region in space.walk() {
                stages.fill(0usize, &region);
            }
        }
        Role::Compute => {
            for region in space.walk() {
                stages.fill(0usize, &region);
                stages.slot_mut(0usize).publish();
                let c_block = c.at(&region);
                stages.consume(0usize, |a_s, b_s| {
                    for cell in &region {
                        let mut c_cell = c_block
                            .at(&cell)
                            .accumulating(REGISTER_BLOCK, Semiring::SUM_PROD);
                        c_cell.mma(&a_s.at(&cell), &b_s.at(&cell));
                    }
                });
            }
        }
    }
}

fn check_ring_matmul(m: usize, n: usize, k: usize, block_k: usize, depth: usize) {
    let client = cubecl::test_device().client();
    let tile = 4usize;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, tile), (N, tile), (K, tile)])
                .walk(&[(M, m / tile), (N, n / tile), (K, block_k / tile)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile, tile])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile, tile])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile, tile])
        .uniform(7, -100.0, 100.0);

    ring_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(&launcher, 1),
        b.arg(&launcher, 1),
        c.arg(&launcher, 1),
        launcher.partitioning_arg(),
        depth,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected = references::tiled_matmul(m, n, k, tile);
    let (_, expected) = TestInput::builder(client, shape![m / tile, n / tile, tile, tile])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3).enforce()
}

/// The two roles meet on a barrier and nowhere else, so a device without one is told so by name
/// rather than handed two loops with no rendezvous between them.
#[test]
fn a_device_without_barriers_refuses_a_walk_filled_by_planes_of_its_own() {
    let (m, n, k, tile) = (8usize, 8usize, 16usize, 4usize);
    let client = cubecl::test_device().client();
    let partitioning = Partitioning::new(
        Space::new(&[(M, m), (N, n), (K, k)]),
        Levels::leaf(&[(M, tile), (N, tile), (K, tile)])
            .walk(&[(M, m / tile), (N, n / tile), (K, 1)])
            .walk_every(&[M, N, K])
            .filled_by(1)
            .build(),
    );
    let concrete = partitioning.space().clone();
    let launched = Launcher::new(&client, partitioning, &concrete, Grid::FromLevels);
    let barriers = client
        .properties()
        .features
        .types
        .opaque
        .contains(&OpaqueType::Barrier);
    match barriers {
        true => assert!(launched.is_ok()),
        false => assert!(matches!(
            launched.err(),
            Some(Refusal::FillersWithoutBarriers { fillers: 1 })
        )),
    }
}

/// The specialized shape runs and is right where no plane is set aside: the compute arm fills
/// and reads its own slot, and the fill arm compiles beside it, entered by nobody.
#[test]
fn a_role_split_walk_with_no_filling_plane_is_the_walk_it_always_was() {
    let (m, n, k, tile) = (8usize, 8usize, 16usize, 4usize);
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, tile), (N, tile), (K, tile)])
                .walk(&[(M, m / tile), (N, n / tile), (K, 1)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile, tile])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile, tile])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile, tile])
        .uniform(7, -100.0, 100.0);

    role_split_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(&launcher, 1),
        b.arg(&launcher, 1),
        c.arg(&launcher, 1),
        launcher.partitioning_arg(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected = references::tiled_matmul(m, n, k, tile);
    let (_, expected) = TestInput::builder(client, shape![m / tile, n / tile, tile, tile])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3).enforce()
}

#[test]
fn a_hand_written_ring_walk_single_buffered() {
    check_ring_matmul(8, 8, 16, 4, 1);
}

#[test]
fn a_hand_written_ring_walk_double_buffered() {
    check_ring_matmul(8, 8, 16, 4, 2);
}

#[test]
fn a_hand_written_ring_walk_triple_buffered_over_an_odd_walk() {
    check_ring_matmul(8, 8, 20, 4, 3);
}
