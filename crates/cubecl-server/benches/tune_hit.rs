//! What a launch pays to find its tuned pick, with each thread driving a device of its own as a
//! model split across devices does.

#[path = "../tests/dummy/mod.rs"]
#[allow(dead_code)]
mod dummy;

use crate::dummy::{DummyDevice, test_client};
use cubecl_common::device::DeviceId;
use cubecl_server::{
    local_tuner,
    tune::{AutotuneKey, LocalTuner, Tunable, TunableSet},
};
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::hint::black_box;
use std::sync::Barrier;
use std::time::Instant;

/// Shaped like a matmul key: a few small integers.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct Key {
    m: u32,
    n: u32,
    k: u32,
}

impl Display for Key {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{}", self.m, self.n, self.k)
    }
}

impl AutotuneKey for Key {}

/// Shaped like burn's `CubeTuneId`: the device and the runtime's name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Id {
    device: DeviceId,
    runtime: &'static str,
}

const KEYS: u32 = 16;
const LAUNCHES: u64 = 1_000_000;

static TUNER: LocalTuner<Key, Id> = local_tuner!("tune_hit");

fn set() -> TunableSet<Key, u32, ()> {
    TunableSet::new_cloning_inputs(|m: &u32| Key {
        m: *m,
        n: 64,
        k: 64,
    })
    .with(Tunable::new("first", |_m: u32| Ok::<(), String>(())))
    .with(Tunable::new("second", |_m: u32| Ok::<(), String>(())))
}

fn id(device: u16) -> Id {
    Id {
        device: DeviceId::new(0, device),
        runtime: "wgpu<spirv>",
    }
}

fn main() {
    #[cfg(persistence)]
    let root = tempfile::tempdir().unwrap();
    #[cfg(persistence)]
    {
        use cubecl_server::config::RuntimeConfig;

        let _ = cubecl_server::config::CubeClRuntimeConfig::get();
        cubecl_environment::environment::set_root(root.path());
    }
    let client = test_client(&DummyDevice);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8)) as u16;
    for device in 0..threads {
        for m in 0..KEYS {
            let tunables = TUNER.init(&id(device), set);
            TUNER.execute(&id(device), &client, tunables, m);
        }
    }

    println!("{LAUNCHES} launches per thread, every key tuned\n");
    for threads in [1, 2, threads] {
        let barrier = Barrier::new(threads as usize);
        let per_launch: Vec<f64> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|device| {
                    let (client, barrier) = (client.clone(), &barrier);
                    scope.spawn(move || {
                        let id = id(device);
                        barrier.wait();
                        let start = Instant::now();
                        for launch in 0..LAUNCHES {
                            let tunables = TUNER.init(&id, set);
                            let m = black_box(launch as u32 % KEYS);
                            TUNER.execute(&id, &client, tunables, m);
                        }
                        start.elapsed().as_nanos() as f64 / LAUNCHES as f64
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mean = per_launch.iter().sum::<f64>() / per_launch.len() as f64;
        println!("{threads} thread(s), a device id each: {mean:>7.1} ns per launch");
    }
}
