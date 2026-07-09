//! CURRENT problem (surfaced by updating to cubecl rev d9960a8).
//!
//! The CPU backend silently drops any launch whose cube has more than
//! `max_units_per_cube` (= number of CPU threads) units. No error is surfaced;
//! the output is simply never written.
//!
//! This is why cubek's `shared_sum` (cube_dim = 32x8 = 256 units) returns 0 in
//! the `reduce::...::reduce_shared::test_shared_sum` tests.
//!
//!     cargo run --release --bin current_shared_sum
//!
//! Backend site: cubecl-cpu/src/compute/stream.rs (the `MaxUnitPerCube` early
//! return swallows the error).

use cubecl::prelude::*;

type R = cubecl::cpu::CpuRuntime;

// Unit 0 writes 42. Nothing else.
#[cube(launch_unchecked)]
fn write_42(output: &mut [f32]) {
    if UNIT_POS == 0 {
        output[0] = 42.0;
    }
}

fn run(client: &ComputeClient<R>, units: u32) -> f32 {
    let out = client.create_from_slice(f32::as_bytes(&[-1.0])); // sentinel
    unsafe {
        write_42::launch_unchecked::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(out.clone(), 1),
        );
    }
    f32::from_bytes(&client.read_one(out).unwrap())[0]
}

fn main() {
    let client = R::client(&Default::default());
    let max = client.properties().hardware.max_units_per_cube;
    println!("max_units_per_cube = {max}\n");

    // A legal cube writes 42. A cube of `max + 1` units is silently dropped:
    // the output keeps its sentinel -1, and no error is reported.
    for units in [max, max + 1] {
        let got = run(&client, units);
        let status = if got == 42.0 { "OK" } else { "*** launch silently dropped ***" };
        println!("cube_dim = {units:>3} units  ->  output = {got}   {status}");
    }
}
