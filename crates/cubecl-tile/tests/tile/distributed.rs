//! Units that take their tiles in turns, however many the launch runs ([`Count::Distributed`]).
//!
//! The partitioning states how many tiles a plane's units share and nothing of how many units
//! there are: the kernel reads that off the launch. So one compiled kernel is right at any unit
//! count, fewer units than tiles (each takes several) or more (some take none), which is what
//! these tests launch it at.
#![allow(non_snake_case)]

use crate::support::{HostData, HostDataType, TestInput};
use cubecl::{prelude::*, zspace::shape};

use cubecl_tile::kind::Boundary;
use cubecl_tile::launch::BoundaryPolicy;
use cubecl_tile::launch::Grid;
use cubecl_tile::*;

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// Rows and columns of the block one unit owns, and the depth one step of its walk reads.
const BLOCK: [usize; 3] = [2, 2, 4];

/// The planes a cube runs, each launched `units` wide.
const PLANES: usize = 2;

/// `c = a · b`, every unit summing its blocks in registers.
#[cube(launch)]
fn distributed_block_matmul<E: Numeric>(
    a: &TileArg<'_>,
    b: &TileArg<'_>,
    c: &TileArg<'_>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(&space);
    let b = b.tile(&space);
    let c = c.tile(&space);
    for cube in &space {
        for plane in cube {
            for unit in plane {
                let (a_unit, b_unit, c_unit) = (a.at(&unit), b.at(&unit), c.at(&unit));
                let sum = c_unit.accumulator::<E>(
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
    }
}

fn reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    let a = |i: usize, p: usize| ((i * k + p) % 7) as f32 - 3.0;
    let b = |p: usize, j: usize| ((p * n + j) % 5) as f32 - 2.0;
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| a(i, p) * b(p, j)).sum()
        })
        .collect()
}

/// `a · b` with each plane's `distributed` blocks along `n` taken in turns by `units` units.
fn run(m: usize, n: usize, k: usize, distributed: usize, units: u32) -> HostData {
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
    // Filled with a value no product here reaches, so a block no unit took shows.
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(dtype)
        .custom(vec![-1e6; m * n])
        .generate_without_host_data();

    let [rows, columns, depth] = BLOCK;
    let partitioning = Partitioning::new(
        Space::new(&[(M, m), (N, n), (K, k)]),
        Levels::leaf(&[(M, rows), (N, columns), (K, depth)])
            .walk_every(&[K])
            .units_distributed(N, distributed)
            .planes(&[(M, PLANES)])
            .cubes(&[M, N])
            .build(),
    );
    // The units are the launch's: the partitioning states how many tiles they share, not how
    // many of them there are.
    let grid = Grid::Stated {
        cube_count: partitioning.cube_count(),
        cube_dim: partitioning.cube_dim(units),
    };
    let space = partitioning.space().clone();
    let launcher = Launcher::new(&client, partitioning, &space, grid).unwrap();

    distributed_block_matmul::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        TileArgLaunch::new(
            a_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, K]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        ),
        TileArgLaunch::new(
            b_handle.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[K, N]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        ),
        TileArgLaunch::new(
            out.clone().binding().into_tensor_arg(),
            dtype,
            1,
            TileSpec::direct(&[M, N]).boundary(BoundaryPolicy::Every(Boundary::Zero)),
        ),
        launcher.partitioning_arg(),
        dtype,
    );

    HostData::from_tensor_handle(&client, out, HostDataType::F32)
}

fn assert_matches(got: &HostData, m: usize, n: usize, k: usize) {
    let want = reference(m, n, k);
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

/// The unit counts of `wanted` whose cube the test device holds: a CPU runtime's cube holds as
/// many units as the host has cores, and a small CI runner has four.
fn fitting(wanted: &[u32]) -> Vec<u32> {
    let max_units = cubecl::test_device()
        .client()
        .properties()
        .hardware
        .max_units_per_cube;
    let fitting: Vec<u32> = wanted
        .iter()
        .copied()
        .filter(|&units| units * PLANES as u32 <= max_units)
        .collect();
    assert!(
        !fitting.is_empty(),
        "no unit count of {wanted:?} fits a cube of {max_units} units"
    );
    fitting
}

/// Twelve blocks a plane, and as many units as fit the launch: one unit takes all twelve, two take
/// six each, four take three each, eight take one or two, and sixteen leave four idle. The same
/// partitioning every time.
#[test]
fn every_block_is_taken_once_at_any_unit_count() {
    let (m, n, k) = (8, 48, 12);
    for units in fitting(&[1, 2, 4, 8, 16]) {
        assert_matches(&run(m, n, k, 12, units), m, n, k);
    }
}

/// A shape that tiles by nothing: the cubes overhang both axes of the output and the walk
/// overhangs `K`, and the units still take every block inside once.
#[test]
fn distributed_blocks_mask_a_ragged_edge() {
    let (m, n, k) = (7, 29, 11);
    for units in fitting(&[2, 4, 16]) {
        assert_matches(&run(m, n, k, 6, units), m, n, k);
    }
}
