//! A loop down to a partitioning's leaves, whatever levels lie between.
//!
//! [`Region::leaves`] walks every level below a region, so a kernel reads the same whether its
//! partitioning stacks no level, one, or several between the region it holds and the leaf.

use super::{Form, implied};
use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::Shape};
use cubecl_tile::*;

const M: Axis = Axis(0);

const CELLS: usize = 6;

/// `out[m] = m + 1` for every leaf, reached from the outermost level through `leaves`.
#[cube(launch)]
fn mark_leaves(out: &mut Tensor<f32>, space: Partitioning) {
    for outer in space {
        for leaf in outer.leaves() {
            let m = leaf.origin(M);
            out[m] = f32::cast_from(m) + 1.0;
        }
    }
}

/// Every cell marked exactly once over `levels`, whose leaf is one cell.
fn assert_every_leaf_visited(levels: Levels) {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(Space::new(&[(M, CELLS)]), levels.build()),
        Form::Static,
    );
    // Unwritten cells keep a value no leaf writes, so a missed leaf fails.
    let out = TestInput::builder(client.clone(), Shape::new([CELLS]))
        .dtype(f32::elem_type_native())
        .custom(vec![-1.0; CELLS])
        .generate_without_host_data();

    mark_leaves::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        out.clone().binding().into_tensor_arg(),
        launcher.partitioning_arg(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    for m in 0..CELLS {
        assert_eq!(got.get_f32(&[m]), m as f32 + 1.0, "cell {m}");
    }
}

#[test]
fn the_outer_level_is_itself_the_leaf() {
    assert_every_leaf_visited(Levels::leaf(&[(M, 1)]).walk(&[(M, CELLS)]));
}

#[test]
fn leaves_walk_the_level_below() {
    assert_every_leaf_visited(Levels::leaf(&[(M, 1)]).walk(&[(M, 2)]).walk(&[(M, 3)]));
}

#[test]
fn leaves_walk_every_level_below() {
    assert_every_leaf_visited(
        Levels::leaf(&[(M, 1)])
            .walk(&[(M, 2)])
            .walk(&[(M, 1)])
            .walk(&[(M, 3)]),
    );
}

/// `out[m] = m + 1` for every plane's origin `m`, reached from the cube through `planes`, or
/// through its walk's when `through_walk`.
#[cube(launch)]
fn mark_planes(out: &mut Tensor<f32>, space: Partitioning, #[comptime] through_walk: bool) {
    for cube in &space {
        if through_walk {
            for plane in cube.walk().planes() {
                let m = plane.origin(M);
                out[m] = f32::cast_from(m) + 1.0;
            }
        } else {
            for plane in cube.planes() {
                let m = plane.origin(M);
                out[m] = f32::cast_from(m) + 1.0;
            }
        }
    }
}

/// The cells `planes` reaches over `levels`, every other left at -1.
fn planes_reached(levels: Levels, through_walk: bool) -> Vec<f32> {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(Space::new(&[(M, CELLS)]), levels.build()),
        Form::Static,
    );
    let out = TestInput::builder(client.clone(), Shape::new([CELLS]))
        .dtype(f32::elem_type_native())
        .custom(vec![-1.0; CELLS])
        .generate_without_host_data();

    mark_planes::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        out.clone().binding().into_tensor_arg(),
        launcher.partitioning_arg(),
        through_walk,
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    (0..CELLS).map(|m| got.get_f32(&[m])).collect()
}

/// A cube's planes are the same however many walks lie between: the walks' first steps. With no
/// walk, each of the three cubes reaches both its planes.
#[test]
fn planes_reach_the_first_step_of_every_walk_above_them() {
    let first_step = [1.0, 2.0, -1.0, -1.0, -1.0, -1.0];
    let one_walk = Levels::leaf(&[(M, 1)])
        .planes(&[(M, 2)])
        .walk(&[(M, 3)])
        .cubes(&[M]);
    assert_eq!(planes_reached(one_walk, false), first_step);
    let two_walks = Levels::leaf(&[(M, 1)])
        .planes(&[(M, 2)])
        .walk(&[(M, 1)])
        .walk(&[(M, 3)])
        .cubes(&[M]);
    assert_eq!(planes_reached(two_walks, false), first_step);
    let no_walk = Levels::leaf(&[(M, 1)]).planes(&[(M, 2)]).cubes(&[M]);
    assert_eq!(
        planes_reached(no_walk, false),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// A walk's planes are those of its first step, the same as the region above it reaches.
#[test]
fn a_walks_planes_are_those_of_its_first_step() {
    let levels = Levels::leaf(&[(M, 1)])
        .planes(&[(M, 2)])
        .walk(&[(M, 3)])
        .cubes(&[M]);
    assert_eq!(
        planes_reached(levels, true),
        [1.0, 2.0, -1.0, -1.0, -1.0, -1.0]
    );
}
