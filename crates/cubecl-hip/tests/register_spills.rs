#![cfg(any(target_os = "linux", target_os = "windows"))]

//! With register spills refused, an autotune whose first candidate spills drops it as it drops
//! any candidate that fails to compile, and picks the one that fits its registers.
//!
//! The configuration is read once for each process, and the test opens a process-wide
//! override, so this file has one test.

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_hip::HipRuntime;
use cubecl_server::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl_server::execution::{ProcessMode, ProcessModeOverride, StatisticsCollector};
use cubecl_server::runtime::Runtime;
use cubecl_server::server::Handle;
use cubecl_server::tune::{LocalTuner, Tunable, TunableSet};
use std::sync::Arc;

/// The values each unit of [`scale_held`] holds: more than a unit's 256 vector registers.
const HELD_VALUES: usize = 320;

/// The units of a cube, enough that each holds values of its own in vector registers.
const UNITS: u32 = 32;

/// Scales [`HELD_VALUES`] values of each unit's own, held in a local array, once per input,
/// every loop over the array unrolled: all of them live in registers across the loop over the
/// inputs, which spills on every AMD part.
// Cube code indexes: it has no iterators to lower.
#[allow(clippy::needless_range_loop)]
#[cube(launch)]
fn scale_held(input: &[f32], output: &mut [f32], #[comptime] n: usize) {
    let mut held = Array::<f32>::new(n);
    #[unroll]
    for i in 0..n {
        held[i] = input[ABSOLUTE_POS * n + i];
    }
    for r in 0..input.len() {
        let factor = input[r];
        #[unroll]
        for i in 0..n {
            held[i] = held[i] * factor + held[(i + 1) % n];
        }
    }
    #[unroll]
    for i in 0..n {
        output[ABSOLUTE_POS * n + i] = held[i];
    }
}

/// Writes 1 to the first value of `output`.
#[cube(launch)]
fn write_one(output: &mut [f32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = 1.0;
    }
}

type Inputs = (Client, Handle, Handle);

/// The autotune of `key` between a candidate that spills, writing 0 from its zero input, and one
/// that writes 1.
fn tune_set(key: String) -> TunableSet<String, Inputs, ()> {
    let values = HELD_VALUES * UNITS as usize;
    let spills = move |(client, input, output): Inputs| -> Result<(), String> {
        // SAFETY: each handle holds `values` values of `f32`.
        let buffer = |handle| unsafe { BufferArg::from_raw_parts(handle, values) };
        scale_held::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(UNITS),
            buffer(input),
            buffer(output),
            HELD_VALUES,
        );
        Ok(())
    };
    let fits = move |(client, _input, output): Inputs| -> Result<(), String> {
        write_one::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_1d(UNITS),
            // SAFETY: the handle holds `values` values of `f32`.
            unsafe { BufferArg::from_raw_parts(output, values) },
        );
        Ok(())
    };
    TunableSet::new_cloning_inputs(move |_: &Inputs| key.clone())
        .with(Tunable::new("spills", spills))
        .with(Tunable::new("fits", fits))
}

#[test]
fn autotune_picks_the_candidate_that_fits_its_registers() {
    let mut config = CubeClRuntimeConfig::default();
    config.compilation.refuse_register_spills = true;
    CubeClRuntimeConfig::set(config);

    static TUNER: LocalTuner<String, String> = LocalTuner::new("register_spills");
    let key = format!(
        "spilling-candidate-{:?}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
    );
    let client = HipRuntime::client(&Default::default());
    let values = HELD_VALUES * UNITS as usize;
    let input = client.create_from_slice(f32::as_bytes(&vec![0.0; values]));
    let output = client.create_from_slice(f32::as_bytes(&vec![7.0; values]));
    let set = Arc::new(tune_set(key));
    let id = String::from("device");
    let inputs: Inputs = (client.clone(), input, output.clone());

    let collector = StatisticsCollector::new();
    let pass = ProcessModeOverride::new(ProcessMode::CompileAndAutotune, &collector);
    TUNER.execute(&id, &client, set.clone(), inputs.clone());
    let _ = client.read_one(client.create_from_slice(f32::as_bytes(&[0.0])));
    core::mem::drop(pass);
    let compilation = collector.statistics().compilation;
    assert!(
        compilation.refused_count >= 1,
        "the spilling candidate was refused: {compilation:?}"
    );

    TUNER.execute(&id, &client, set, inputs);
    let actual = client.read_one(output).expect("the pick runs");
    assert_eq!(
        f32::from_bytes(&actual)[0],
        1.0,
        "the pick is the candidate that fits"
    );
}
