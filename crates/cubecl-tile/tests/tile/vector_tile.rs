//! A kernel reads where each load of a window starts off the load's own tile: a plain run along
//! the innermost dimension, or an NVFP4 rectangle spanning two dimensions.

use cubecl::prelude::*;
use cubecl_tile::{Dimension, Space, layout::VectorTile};

const K: Dimension = Dimension(0);
const N: Dimension = Dimension(1);

/// Each unit writes the `(k, n)` its load starts at.
#[cube(launch)]
fn load_starts(out: &mut [u32], #[comptime] load: VectorTile, #[comptime] space: Space) {
    let start = load.start(UNIT_POS, &space);
    out[(UNIT_POS * 2) as usize] = start.at(0);
    out[(UNIT_POS * 2 + 1) as usize] = start.at(1);
}

fn starts(load: VectorTile, loads: u32) -> Vec<u32> {
    starts_over(load, loads, Space::new(&[(K, 32), (N, 4)]))
}

fn starts_over(load: VectorTile, loads: u32, space: Space) -> Vec<u32> {
    let client = cubecl::test_device().client();
    let out = client.empty(2 * loads as usize * size_of::<u32>());
    load_starts::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(loads),
        unsafe { BufferArg::from_raw_parts(out.clone(), 2 * loads as usize) },
        load,
        space,
    );
    u32::from_bytes(&client.read_one_unchecked(out)).to_vec()
}

/// A 32 by 4 window read in 16 by 2 loads is four loads, starting two columns apart along `N`
/// and 16 values apart along `K`; read in runs of 4 along `N`, it is 32 loads, one per `k`.
#[test]
fn a_load_starts_a_whole_tile_apart_along_each_dimension() {
    let rectangle = VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32).unwrap();
    assert_eq!(starts(rectangle, 4), vec![0, 0, 0, 2, 16, 0, 16, 2]);

    let run = VectorTile::new(&[], N, 4).unwrap();
    let expected: Vec<u32> = (0..32).flat_map(|k| [k, 0]).collect();
    assert_eq!(starts(run, 32), expected);
}

/// A buffer stored in tiles above one load is read a stored tile at a time: the kernel's
/// neighbouring loads step along `K` through one 64-deep tile, then to the tile's next columns,
/// before the next tile along `K`.
#[test]
fn neighbouring_loads_step_through_one_stored_tile_first() {
    let stored = [(K, 8), (K, 2), (N, 2), (K, 4), (N, 2)];
    let load = VectorTile::new(&stored, N, 32).unwrap();
    let starts = starts_over(load, 16, Space::new(&[(K, 128), (N, 4)]));
    let expected: Vec<u32> = [
        (0, 0),
        (16, 0),
        (32, 0),
        (48, 0),
        (0, 2),
        (16, 2),
        (32, 2),
        (48, 2),
        (64, 0),
        (80, 0),
        (96, 0),
        (112, 0),
        (64, 2),
        (80, 2),
        (96, 2),
        (112, 2),
    ]
    .into_iter()
    .flat_map(|(k, n)| [k, n])
    .collect();
    assert_eq!(starts, expected);
}
