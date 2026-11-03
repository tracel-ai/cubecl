//! A kernel reads where each load of a window starts off the load's own tile: a plain run along
//! the innermost axis, or an NVFP4 rectangle spanning two axes.

use cubecl::prelude::*;
use cubecl_tile::{Axis, Space, layout::VectorTile};

const K: Axis = Axis(0);
const N: Axis = Axis(1);

/// Each unit writes the `(k, n)` its load starts at.
#[cube(launch)]
fn load_starts(out: &mut [u32], #[comptime] load: VectorTile, #[comptime] space: Space) {
    let start = load.start(UNIT_POS, &space);
    out[(UNIT_POS * 2) as usize] = start.at(0);
    out[(UNIT_POS * 2 + 1) as usize] = start.at(1);
}

fn starts(load: VectorTile, loads: u32) -> Vec<u32> {
    let client = cubecl::test_device().client();
    let out = client.empty(2 * loads as usize * size_of::<u32>());
    load_starts::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(loads),
        unsafe { BufferArg::from_raw_parts(out.clone(), 2 * loads as usize) },
        load,
        Space::new(&[(K, 32), (N, 4)]),
    );
    u32::from_bytes(&client.read_one_unchecked(out)).to_vec()
}

/// A 32 by 4 window read in 16 by 2 loads is four loads, starting two columns apart along `N`
/// and 16 values apart along `K`; read in runs of 4 along `N`, it is 32 loads, one per `k`.
#[test]
fn a_load_starts_a_whole_tile_apart_along_each_axis() {
    let rectangle = VectorTile::new(&[(K, 8), (K, 2), (N, 2)], N, 32).unwrap();
    assert_eq!(starts(rectangle, 4), vec![0, 0, 0, 2, 16, 0, 16, 2]);

    let run = VectorTile::new(&[], N, 4).unwrap();
    let expected: Vec<u32> = (0..32).flat_map(|k| [k, 0]).collect();
    assert_eq!(starts(run, 32), expected);
}
