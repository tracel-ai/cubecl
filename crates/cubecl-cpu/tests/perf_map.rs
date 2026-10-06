//! `CUBECL_JIT_SYMBOLS=perf` writes one perf map line for each kernel.
//!
//! The variable is read once for each process, so this file has one test.
#![cfg(target_os = "linux")]

use cubecl_core as cubecl;
use cubecl_core::ir::settings::DebugInfo;
use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_server::config::compilation::PROFILE_DEBUG_INFO;
use cubecl_server::runtime::Runtime;

#[cube(launch)]
fn perf_map_add(output: &mut [f32]) {
    output[ABSOLUTE_POS] += 1.0;
}

#[cube(launch)]
fn perf_map_mul(output: &mut [f32]) {
    output[ABSOLUTE_POS] *= 2.0;
}

#[test]
fn each_kernel_has_a_perf_map_line() {
    // SAFETY: no other thread of this test binary reads the environment yet.
    unsafe { std::env::set_var("CUBECL_JIT_SYMBOLS", "perf") };
    let client = CpuRuntime::client(&Default::default());
    let output = client.create_from_slice(f32::as_bytes(&[1.0; 4]));
    let arg = || unsafe { BufferArg::from_raw_parts(output.clone(), 4) };
    perf_map_add::launch(&client, CubeCount::new_single(), CubeDim::new_1d(4), arg());
    perf_map_mul::launch(&client, CubeCount::new_single(), CubeDim::new_1d(4), arg());
    client.read_one(output).unwrap();

    let path = format!("/tmp/perf-{}.map", std::process::id());
    let map = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    if PROFILE_DEBUG_INFO == DebugInfo::None {
        // The files need debug data in the cargo profile.
        assert!(map.is_empty(), "{map}");
        return;
    }

    for kernel in ["perf_map_add", "perf_map_mul"] {
        let lines = map
            .lines()
            .filter(|line| line.contains(kernel))
            .collect::<Vec<_>>();
        assert_eq!(lines.len(), 1, "{kernel} in:\n{map}");
        let fields = lines[0].split(' ').collect::<Vec<_>>();
        assert_eq!(fields.len(), 3, "{}", lines[0]);
        assert_ne!(u64::from_str_radix(fields[0], 16).unwrap(), 0, "address");
        assert_ne!(u64::from_str_radix(fields[1], 16).unwrap(), 0, "size");
    }
}
