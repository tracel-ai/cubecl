//! OLD problem (pre-existing; the one the interpolate crate works around by
//! refusing the shared-memory strategy on CPU).
//!
//! When a launch dispatches MORE THAN ONE cube, all cubes share ONE shared-memory
//! buffer. `sync_cube` only synchronizes within a cube, so cubes that run
//! concurrently race and corrupt each other. Single-cube shared memory is fine —
//! the bug needs cube_count > 1.
//!
//! This is why cubek's interpolate `*_shared_memory_*` tests fail on CPU: they
//! all dispatch >= 2 cubes (one per output tile / batch).
//!
//!     cargo run --release --bin old_interpolate
//!
//! Backend site: cubecl-cpu/src/compiler/mlir_data.rs `MlirData::new` allocates
//! the shared buffer once per launch into one `Arc<SharedMlirData>` that the
//! threadpool clones to every unit of every cube — there is no per-cube buffer.

use cubecl::prelude::*;

type R = cubecl::cpu::CpuRuntime;

// Each cube fills its own shared buffer with (CUBE_POS + 1), syncs, then every
// unit sums the buffer. Correct result for cube c is n * (c + 1).
#[cube(launch_unchecked)]
fn cube_sum(output: &mut [f32], #[comptime] n: u32) {
    let mut smem = Shared::new_slice(n as usize);
    smem[UNIT_POS as usize] = f32::cast_from(CUBE_POS) + 1.0;
    sync_cube();
    let mut acc = 0.0;
    for i in 0..n {
        acc += smem[i as usize];
    }
    output[(CUBE_POS as usize) * (n as usize) + UNIT_POS as usize] = acc;
}

fn main() {
    let client = R::client(&Default::default());
    let (n, cubes) = (8u32, 8u32); // 8 units per cube, 8 cubes

    let out = client.empty((n * cubes) as usize * size_of::<f32>());
    unsafe {
        cube_sum::launch_unchecked::<R>(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(n),
            BufferArg::from_raw_parts(out.clone(), (n * cubes) as usize),
            n,
        );
    }
    let got = f32::from_bytes(&client.read_one(out).unwrap()).to_vec();

    // One value per cube (they should all be identical within a cube).
    let per_cube: Vec<f32> = (0..cubes).map(|c| got[(c * n) as usize]).collect();
    let expected: Vec<f32> = (1..=cubes).map(|c| (n * c) as f32).collect();
    let ok = per_cube == expected;

    println!("expected per-cube sums = {expected:?}");
    println!("got      per-cube sums = {per_cube:?}");
    println!(
        "\n{}  (rerun a few times: the wrong values change => it's a race)",
        if ok { "OK" } else { "*** WRONG: cubes share one shared buffer and race ***" }
    );
}
