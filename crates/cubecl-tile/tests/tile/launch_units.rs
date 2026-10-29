//! A unit level spanning the launch ([`Levels::launch_units`]): the outermost level, with no cube
//! or plane level above it, one tile to each unit of the whole launch, the units grouped in cubes
//! of a [`UnitCube`], so a kernel walks them as `for unit in &partitioning`.
//!
//! Every case adds its products atomically into a zeroed output, so a tile no unit took reads
//! zero and a tile two units took reads twice its product: a unit past the last tile walks one
//! past the grid, which the binding masks.
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput, ValidationResult};
use cubecl::{
    features::AtomicUsage,
    ir::{ElemType, FloatKind, Type},
    prelude::*,
    zspace::shape,
};

use cubecl_tile::kind::{Boundary, Write};
use cubecl_tile::launch::{BoundaryPolicy, Buffered, Destination, DestinationLaunch, Grid};
use cubecl_tile::space::UnitCube;
use cubecl_tile::*;

use super::launch_planes::holds_a_cube;
use super::{Form, kernel_form};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
/// A batch the inputs do not span: every batch's output is the same product.
const B: Axis = Axis(3);

/// Rows, columns and depth of the block one unit owns.
const BLOCK: [usize; 3] = [2, 2, 4];

/// `c += a · b`, every unit of the launch its block.
#[cube(launch)]
fn launch_unit_matmul<O: Destination>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &O::Arg<f32>,
    space: Partitioning,
) {
    let (a, b) = (a.tile(&space), b.tile(&space));
    let c = O::tile::<f32, Const<1>>(c, &space);
    for unit in &space {
        let (a_unit, b_unit, c_unit) = (a.at(&unit), b.at(&unit), c.at(&unit));
        let sum = c_unit.accumulator::<f32>(
            &a_unit,
            &b_unit,
            comptime!(Instruction::Registers {
                config: RegisterBlock::new(BLOCK[0] * BLOCK[1])
            }),
            Semiring::SUM_PROD,
        );
        for step in &unit {
            let mut sum_step = sum.at(&step);
            sum_step.mma(&a_unit.at(&step), &b_unit.at(&step));
        }
        sum.drained_into(&c_unit);
    }
}

/// How many tiles every unit of the launch walks, at the unit's place in the launch.
#[cube(launch)]
fn count_walks(walks: &mut Tensor<f32>, space: Partitioning) {
    let mut taken = 0.0f32;
    for _unit in &space {
        taken += 1.0;
    }
    let cube = CUBE_POS_X + CUBE_POS_Y * CUBE_COUNT_X;
    walks[(cube * CUBE_DIM + UNIT_POS) as usize] = taken;
}

/// The levels a unit takes its `2 × 2` block at, its units in cubes of `cube`, the units handing
/// out `reversed` from the last tile to the first.
fn levels(cube: UnitCube, reversed: &[Axis]) -> Vec<Level> {
    let [rows, columns, depth] = BLOCK;
    let outermost = Levels::leaf(&[(M, rows), (N, columns), (K, depth)])
        .walk_every(&[K])
        .launch_units(&[M, N, B], cube);
    reversed
        .iter()
        .fold(outermost, |levels, &axis| levels.reversed(axis))
        .build()
}

/// An `m × n × k` product, written into each of `batches` batches of the output.
#[derive(Clone, Copy, Debug)]
struct Problem {
    m: usize,
    n: usize,
    k: usize,
    batches: usize,
}

impl Problem {
    const fn new(m: usize, n: usize, k: usize, batches: usize) -> Self {
        Self { m, n, k, batches }
    }
}

/// Problems whose edges cut the blocks short, one and three batches.
const PROBLEMS: [Problem; 4] = [
    Problem::new(8, 16, 8, 1),
    Problem::new(7, 29, 11, 1),
    Problem::new(13, 5, 6, 3),
    Problem::new(4, 4, 4, 3),
];

/// Cubes of one plane of one unit, a plane of two, and three planes of two.
const CUBES: [UnitCube; 3] = [
    UnitCube {
        plane_width: 1,
        planes: 1,
    },
    UnitCube {
        plane_width: 2,
        planes: 1,
    },
    UnitCube {
        plane_width: 2,
        planes: 3,
    },
];

fn lhs(i: usize, p: usize, k: usize) -> f32 {
    ((i * k + p) % 7) as f32 - 3.0
}

fn rhs(p: usize, j: usize, n: usize) -> f32 {
    ((p * n + j) % 5) as f32 - 2.0
}

/// Whether the test device adds `f32`s atomically, which every case's output takes.
fn adds_atomically(client: &cubecl::client::Client) -> bool {
    let usage = client
        .properties()
        .atomic_type_usage(Type::atomic(ElemType::Float(FloatKind::F32)));
    let adds = usage.contains(AtomicUsage::Add);
    if !adds {
        ValidationResult::Skipped("device has no f32 atomic add".to_string()).enforce();
    }
    adds
}

/// `problem`'s product, every unit of the launch a block; `None` where the device cannot add into
/// the output or hold its cube.
fn run(problem: Problem, cube: UnitCube, reversed: &[Axis], form: Form<'_>) -> Option<HostData> {
    let Problem { m, n, k, batches } = problem;
    let client = cubecl::test_device().client();
    if !adds_atomically(&client) {
        return None;
    }
    let dtype = f32::elem_type_native();
    let a: Vec<f32> = (0..m * k).map(|i| lhs(i / k, i % k, k)).collect();
    let b: Vec<f32> = (0..k * n).map(|i| rhs(i / n, i % n, n)).collect();
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(dtype)
        .custom(a)
        .generate_with_f32_host_data();
    let (b_handle, _) = TestInput::builder(client.clone(), shape![k, n])
        .dtype(dtype)
        .custom(b)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![batches, m, n])
        .dtype(dtype)
        .custom(vec![0.0; batches * m * n])
        .generate_without_host_data();

    let partitioning = Partitioning::new(
        Space::new(&[(B, batches), (M, m), (N, n), (K, k)]),
        levels(cube, reversed),
    );
    if !holds_a_cube(&client, &partitioning) {
        return None;
    }
    let grid = Grid::Stated {
        cube_count: partitioning.cube_count(),
        cube_dim: partitioning.cube_dim(32),
    };
    let space = partitioning.space().clone();
    let kernel = kernel_form(&partitioning, form);
    let launcher = Launcher::new(&client, kernel, &space, grid).unwrap();
    let policy = BoundaryPolicy::Every(Boundary::Zero);
    let bound = launcher
        .arg(out.clone().binding(), dtype)
        .axes(&[B, M, N])
        .build()
        .unwrap();
    let a_arg = TileArgLaunch::new(
        a_handle.binding().into_tensor_arg(),
        dtype,
        1,
        TileSpec::direct(&[M, K]).boundary(policy),
    );
    let b_arg = TileArgLaunch::new(
        b_handle.binding().into_tensor_arg(),
        dtype,
        1,
        TileSpec::direct(&[K, N]).boundary(policy),
    );
    launch_unit_matmul::launch::<Buffered>(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_arg,
        b_arg,
        Buffered::arg((bound, Write::Accumulate)),
        launcher.partitioning_arg(),
    );
    Some(HostData::from_tensor_handle(
        &client,
        out,
        HostDataType::F32,
    ))
}

/// Every cell of every batch of the output is its product once: no tile left out, none taken
/// twice.
fn assert_every_tile_once(got: &HostData, problem: Problem, what: &str) {
    let Problem { m, n, k, batches } = problem;
    for batch in 0..batches {
        for i in 0..m {
            for j in 0..n {
                let want: f32 = (0..k).map(|p| lhs(i, p, k) * rhs(p, j, n)).sum();
                assert_eq!(
                    got.get_f32(&[batch, i, j]),
                    want,
                    "{what}: at ({batch}, {i}, {j})"
                );
            }
        }
    }
}

/// Every unit of the launch takes one block, in cubes of one unit, of a plane, and of several
/// planes, the last cube's units past the last tile taking none.
#[test]
fn the_launch_units_take_every_tile_once() {
    for cube in CUBES {
        for problem in PROBLEMS {
            if let Some(got) = run(problem, cube, &[], Form::Static) {
                assert_every_tile_once(&got, problem, &format!("{cube:?}"));
            }
        }
    }
}

/// The units reversed along their axes, over static extents and runtime ones.
#[test]
fn reversed_launch_units_take_every_tile_once() {
    let cube = CUBES[2];
    for reversed in [&[M][..], &[N], &[M, N, B]] {
        for form in [Form::Static, Form::Dynamic] {
            for problem in PROBLEMS {
                if let Some(got) = run(problem, cube, reversed, form) {
                    assert_every_tile_once(&got, problem, &format!("{reversed:?}, {form:?}"));
                }
            }
        }
    }
}

/// The cubes the levels ask for: every tile's unit, a cube's worth at a time, in the cube the
/// level states whatever the plane size the launch asks it at; a unit asks for no units of its
/// own.
#[test]
fn the_launch_holds_every_unit() {
    // Blocks of 2 × 2: 4 along m, 15 along n.
    let partitioning = Partitioning::new(
        Space::new(&[(B, 1), (M, 7), (N, 29), (K, 11)]),
        levels(CUBES[2], &[]),
    );
    let CubeCount::Static(x, y, z) = partitioning.cube_count() else {
        panic!("a static space spawns a static grid");
    };
    assert_eq!((x, y, z), (10, 1, 1));
    assert_eq!(partitioning.cube_dim(32), CubeDim::new_2d(2, 3));
    assert_eq!(partitioning.planes_per_cube(), 3);
    assert_eq!(partitioning.units(), 1);
    assert_eq!(partitioning.unit_cube(), Some(CUBES[2]));
}

/// Every unit the launch spawns walks exactly one tile, a unit past the last tile one past the
/// grid.
#[test]
fn every_unit_of_the_launch_walks_one_tile() {
    let client = cubecl::test_device().client();
    for cube in CUBES {
        for problem in PROBLEMS {
            let Problem { m, n, k, batches } = problem;
            let partitioning = Partitioning::new(
                Space::new(&[(B, batches), (M, m), (N, n), (K, k)]),
                levels(cube, &[]),
            );
            if !holds_a_cube(&client, &partitioning) {
                continue;
            }
            let cube_count = partitioning.cube_count();
            let CubeCount::Static(x, y, z) = cube_count else {
                panic!("a static space spawns a static grid");
            };
            let units = (x * y * z) as usize * cube.units();
            let walks = TestInput::builder(client.clone(), shape![units])
                .dtype(f32::elem_type_native())
                .custom(vec![0.0; units])
                .generate_without_host_data();
            let grid = Grid::Stated {
                cube_count,
                cube_dim: partitioning.cube_dim(32),
            };
            let space = partitioning.space().clone();
            let launcher = Launcher::new(&client, partitioning, &space, grid).unwrap();
            count_walks::launch(
                &client,
                launcher.cube_count(),
                launcher.cube_dim(),
                walks.clone().binding().into_tensor_arg(),
                launcher.partitioning_arg(),
            );
            let got = HostData::from_tensor_handle(&client, walks, HostDataType::F32);
            for unit in 0..units {
                assert_eq!(
                    got.get_f32(&[unit]),
                    1.0,
                    "{cube:?} on {problem:?}: unit {unit} of {units}"
                );
            }
        }
    }
}

/// A plane level beneath the launch's units would take a tile the units already split.
#[test]
#[should_panic(expected = "no plane level holds them")]
fn no_plane_level_sits_beneath_the_launch_units() {
    Partitioning::new(
        Space::new(&[(M, 8), (N, 8)]),
        Levels::leaf(&[(M, 2), (N, 2)])
            .planes(&[(M, 2)])
            .launch_units(&[M, N], CUBES[1])
            .build(),
    );
}
