//! End-to-end proof that the LLVM backend produces a working kernel.
//!
//! Requires an AMD GPU and `--features llvm`.

#![cfg(feature = "llvm")]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_hip::HipRuntime;

#[cube(launch)]
pub fn write_absolute_pos(output: &mut [u32]) {
    output[ABSOLUTE_POS] = ABSOLUTE_POS as u32;
}

/// Encodes all three `CubeCount*` axes into one value with distinct decimal weights,
/// so a swapped or misread packet offset shows up as a different number.
#[cube(launch)]
pub fn write_cube_counts(output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = CUBE_COUNT_X + CUBE_COUNT_Y * 100 + CUBE_COUNT_Z * 10000;
    }
}

const N: usize = 256;
const CUBE_DIM: u32 = 64;

#[test]
#[ignore = "requires an AMD GPU"]
fn llvm_backend_runs_a_kernel() {
    // Must be set before the first `client()` call: the context, and with it
    // the compiler choice and cache namespace, is built once per process.
    unsafe { std::env::set_var("CUBECL_HIP_COMPILER", "llvm") };

    let client = HipRuntime::client(&Default::default());
    let handle = client.empty(N * core::mem::size_of::<u32>());

    write_absolute_pos::launch::<HipRuntime>(
        &client,
        CubeCount::Static(N as u32 / CUBE_DIM, 1, 1),
        CubeDim::new_3d(CUBE_DIM, 1, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), N) },
    );

    let bytes = client.read_one(handle).unwrap();
    let got = u32::from_bytes(&bytes).to_vec();

    let expected: Vec<u32> = (0..N as u32).collect();
    assert_eq!(got, expected);
}

/// Exercises the dispatch-packet reads behind `CubeCount*`, which need a genuinely 3-D,
/// asymmetric grid to be covered at all. From `entrypoint.rs`'s `absolute_pos`:
///
/// ```text
/// units_x = cube_count_x * cube_dim_x    // packet offset 12
/// units_y = cube_count_y * cube_dim_y    // packet offset 16
/// ABSOLUTE_POS = abs_z*units_x*units_y + abs_y*units_x + abs_x
/// ```
///
/// `units_x` multiplies `abs_y`, so any 2-D grid covers offset 12. `units_y` multiplies
/// only `abs_z`, so with a 2-D grid (`abs_z == 0`) a wrong offset 16 is invisible — hence
/// the nonzero `cube_count_z`. The counts must also differ from each other: with
/// `(2,2,2)`, swapping offsets 16 and 20 reads the same value either way. Verified by
/// temporarily swapping `GRID_SIZE_Y_OFFSET`/`GRID_SIZE_Z_OFFSET` in `builtins.rs` —
/// green with `(2,2,2)`, red with the `(2,3,4)` used here.
///
/// Offset 20 is not reachable through `ABSOLUTE_POS` at all (`absolute_pos` never takes
/// `cube_count_z`); `llvm_backend_reads_cube_counts` covers it.
///
/// With `cube_count = (2,3,4)`, `cube_dim = (64,1,1)`: `units_x = 128`, `units_y = 3`, so
/// `ABSOLUTE_POS = abs_z*384 + abs_y*128 + abs_x`, a bijection onto `0..1536`. A wrong
/// offset 16 collapses distinct indices onto the same slot.
#[test]
#[ignore = "requires an AMD GPU"]
fn llvm_backend_handles_a_3d_grid() {
    unsafe { std::env::set_var("CUBECL_HIP_COMPILER", "llvm") };

    const N_3D: usize = 1536;

    let client = HipRuntime::client(&Default::default());
    let handle = client.empty(N_3D * core::mem::size_of::<u32>());

    write_absolute_pos::launch::<HipRuntime>(
        &client,
        CubeCount::Static(2, 3, 4),
        CubeDim::new_3d(CUBE_DIM, 1, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), N_3D) },
    );

    let bytes = client.read_one(handle).unwrap();
    let got = u32::from_bytes(&bytes).to_vec();

    let expected: Vec<u32> = (0..N_3D as u32).collect();
    assert_eq!(
        got, expected,
        "3-D grid: CubeCount* from dispatch.ptr is wrong"
    );
}

/// Covers packet offset 20 (`grid_size_z`), which `ABSOLUTE_POS` can never reach.
///
/// HIP launches `grid_size = cube_count * cube_dim`, so `grid_size_n / cube_dim_n`
/// recovers `(2,3,4)` and the expected encoding is `2 + 3*100 + 4*10000 = 40302`.
/// Distinct weights mean any swapped pair of offsets yields a visibly wrong number.
///
/// The division itself is only covered on x: `cube_dim_y` and `cube_dim_z` are `1` here,
/// so those divisions are no-ops. A bug in the shared `cube_count_component` helper,
/// used by all three axes, would still be caught via x.
#[test]
#[ignore = "requires an AMD GPU"]
fn llvm_backend_reads_cube_counts() {
    unsafe { std::env::set_var("CUBECL_HIP_COMPILER", "llvm") };

    let client = HipRuntime::client(&Default::default());
    let handle = client.empty(core::mem::size_of::<u32>());

    write_cube_counts::launch::<HipRuntime>(
        &client,
        CubeCount::Static(2, 3, 4),
        CubeDim::new_3d(CUBE_DIM, 1, 1),
        unsafe { BufferArg::from_raw_parts(handle.clone(), 1) },
    );

    let bytes = client.read_one(handle).unwrap();
    let got = u32::from_bytes(&bytes).to_vec();

    assert_eq!(
        got,
        vec![40302],
        "CUBE_COUNT_X + CUBE_COUNT_Y*100 + CUBE_COUNT_Z*10000: dispatch-packet offsets wrong"
    );
}
