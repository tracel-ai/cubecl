//! Each process measures one cube size to exclude pool growth history.
#![cfg(any(target_os = "linux", target_os = "macos"))]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_cpu::CpuRuntime;
use cubecl_environment::future::block_on;
use cubecl_server::runtime::Runtime;
use std::time::Instant;

#[cube(launch)]
fn barrier_probe(output: &mut [u32], #[comptime] rounds: u32) {
    let mut shared = Shared::new_slice(1usize);
    for round in 0..rounds {
        if UNIT_POS == 0 {
            shared[0] = 42u32 + round;
        }
        sync_cube();
        output[UNIT_POS as usize] = shared[0];
        if comptime!(rounds > 1) {
            sync_cube();
        }
    }
}

fn setting(name: &str, default: u32) -> u32 {
    std::env::var(name).map_or(default, |value| value.parse().expect("positive integer"))
}

fn usage() -> libc::rusage {
    let mut usage = std::mem::MaybeUninit::uninit();
    // SAFETY: getrusage initializes the entire output on success; the pointer is writable.
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },
        0
    );
    unsafe { usage.assume_init() }
}

fn cpu_micros(usage: &libc::rusage) -> i128 {
    (i128::from(usage.ru_utime.tv_sec) + i128::from(usage.ru_stime.tv_sec)) * 1_000_000
        + i128::from(usage.ru_utime.tv_usec)
        + i128::from(usage.ru_stime.tv_usec)
}

/// Example: `CUBECL_BARRIER_UNITS=64 cargo test -p cubecl-cpu --release --test barrier_latency
/// -- --ignored --nocapture --test-threads=1`. CPU time includes the host and all runtime threads.
#[test]
#[ignore = "manual benchmark; emits raw samples without host-dependent timing assertions"]
fn barrier_latency() {
    let units = setting("CUBECL_BARRIER_UNITS", 64);
    let rounds = setting("CUBECL_BARRIER_ROUNDS", 1);
    let launches = setting("CUBECL_BARRIER_LAUNCHES", if units <= 8 { 200 } else { 10 });
    let samples = setting("CUBECL_BARRIER_SAMPLES", 7);
    assert!(units > 0 && rounds > 0 && launches > 0 && samples > 0);
    let client = CpuRuntime::client(&Default::default());
    let output = client.empty(units as usize * size_of::<u32>());
    let launch = || {
        barrier_probe::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(units),
            unsafe { BufferArg::from_raw_parts(output.clone(), units as usize) },
            rounds,
        );
    };
    for _ in 0..3 {
        launch();
        block_on(client.sync()).unwrap();
    }
    println!("\nBARRIER_CSV,units,rounds,sample,launches,wall_ns,cpu_us,voluntary,involuntary");
    for sample in 0..samples {
        let before = usage();
        let start = Instant::now();
        for _ in 0..launches {
            launch();
        }
        block_on(client.sync()).unwrap();
        let wall_ns = start.elapsed().as_nanos();
        let after = usage();
        let cpu_us = cpu_micros(&after) - cpu_micros(&before);
        let voluntary = after.ru_nvcsw - before.ru_nvcsw;
        let involuntary = after.ru_nivcsw - before.ru_nivcsw;
        println!(
            "BARRIER_CSV,{units},{rounds},{sample},{launches},{wall_ns},{cpu_us},{voluntary},{involuntary}"
        );
    }
    let bytes = client.read_one(output).unwrap();
    assert_eq!(u32::from_bytes(&bytes), vec![41 + rounds; units as usize]);
}
