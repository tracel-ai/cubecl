//! Storage barriers on the CPU runtime.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_server::runtime::Runtime;

#[cube(launch)]
fn storage_exchange(output: &mut [u32], #[comptime] system: bool) {
    let base = CUBE_POS * CUBE_DIM as usize;
    let unit = UNIT_POS as usize;
    output[base + unit] = UNIT_POS + 11;
    if system {
        sync_system();
    } else {
        sync_storage();
    }
    let neighbor = output[base + (unit ^ 1)];
    if system {
        sync_system();
    } else {
        sync_storage();
    }
    output[base + unit] = neighbor;
}

#[test]
fn storage_and_system_barriers_exchange_global_writes() {
    let client = CpuRuntime::client(&Default::default());
    for system in [false, true] {
        let output = client.create_from_slice(u32::as_bytes(&[0; 4]));
        storage_exchange::launch(
            &client,
            CubeCount::new_1d(2),
            CubeDim::new_1d(2),
            unsafe { BufferArg::from_raw_parts(output.clone(), 4) },
            system,
        );
        let bytes = client.read_one(output).unwrap();
        assert_eq!(u32::from_bytes(&bytes), &[12, 11, 12, 11]);
    }
}
