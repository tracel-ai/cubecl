//! Reusable cube barriers, including launches with more units than host CPUs.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_server::runtime::Runtime;

// Force the debugger's RuntimeDyld linker even in release; the other kernels use JITLink.
#[cube(launch, debug_symbols)]
fn exchange_rounds(output: &mut [u32], rounds: u32, seed: u32, #[comptime] units: usize) {
    let mut shared = Shared::new_slice(units);
    let unit = UNIT_POS as usize;
    let cube = CUBE_POS as u32;
    for round in 0..rounds {
        shared[unit] = seed + (cube * rounds + round) * CUBE_DIM + UNIT_POS;
        sync_cube();
        output[((cube * rounds + round) * CUBE_DIM + UNIT_POS) as usize] =
            shared[(unit + 1) % CUBE_DIM as usize];
        // No unit may overwrite its slot before all peers have read this round.
        sync_cube();
    }
}

#[test]
fn repeated_barriers_publish_shared_writes_across_cubes_and_launches() {
    let client = CpuRuntime::client(&Default::default());
    let oversubscribed = 64.max(std::thread::available_parallelism().unwrap().get() as u32 + 1);
    let rounds = 8;
    let cubes = 4;

    for dim in [
        CubeDim::new_single(),
        CubeDim::new_1d(2),
        CubeDim::new_1d(oversubscribed),
        CubeDim::new_2d(8, 8),
        CubeDim::new_3d(4, 4, 4),
    ] {
        let units = dim.num_elems();
        let count = (cubes * rounds * units) as usize;
        let output = client.empty(count * size_of::<u32>());
        for seed in [0, 10_000, 20_000] {
            exchange_rounds::launch(
                &client,
                CubeCount::Static(2, 1, 2),
                dim,
                unsafe { BufferArg::from_raw_parts(output.clone(), count) },
                rounds,
                seed,
                units as usize,
            );
            let bytes = client.read_one(output.clone()).unwrap();
            let expected: Vec<_> = (0..cubes * rounds)
                .flat_map(|round| {
                    (0..units).map(move |unit| seed + round * units + (unit + 1) % units)
                })
                .collect();
            assert_eq!(u32::from_bytes(&bytes), expected, "{dim:?}, seed {seed}");
        }
    }
}

#[cube(launch)]
fn exchange_global(buffer: &mut [u32], output: &mut [u32], rounds: u32) {
    let unit = UNIT_POS as usize;
    for round in 0..rounds {
        buffer[unit] = round * CUBE_DIM + UNIT_POS;
        sync_cube();
        // Consecutive barriers must each be reached by every unit.
        sync_cube();
        output[round as usize * CUBE_DIM as usize + unit] = buffer[(unit + 1) % CUBE_DIM as usize];
        sync_cube();
    }
}

#[test]
fn barriers_without_shared_memory_publish_global_writes() {
    let client = CpuRuntime::client(&Default::default());
    let units = 64;
    let rounds = 16;
    let buffer = client.empty(units * size_of::<u32>());
    let output = client.empty(units * rounds * size_of::<u32>());
    exchange_global::launch(
        &client,
        CubeCount::new_single(),
        CubeDim::new_1d(units as u32),
        unsafe { BufferArg::from_raw_parts(buffer, units) },
        unsafe { BufferArg::from_raw_parts(output.clone(), units * rounds) },
        rounds as u32,
    );
    let bytes = client.read_one(output).unwrap();
    let expected: Vec<_> = (0..rounds)
        .flat_map(|round| (0..units).map(move |unit| (round * units + (unit + 1) % units) as u32))
        .collect();
    assert_eq!(u32::from_bytes(&bytes), expected);
}
