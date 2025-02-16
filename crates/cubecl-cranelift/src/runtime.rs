use std::mem::MaybeUninit;

use cubecl_core::{
    ir::{Elem, FloatKind},
    Feature, FeatureSet, Properties, Runtime,
};
use cubecl_runtime::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    memory_management::dynamic::{DynamicMemoryManagement, DynamicMemoryManagementOptions},
    ComputeRuntime,
};

use crate::{
    compiler::CraneLiftCompiler,
    compute::{CraneLiftContext, CraneLiftServer, CraneLiftStorage},
    device::CraneLiftDevice,
};

#[derive(Debug)]
pub struct CraneLiftRuntime;

static RUNTIME: ComputeRuntime<CraneLiftDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

const MEMORY_OFFSET_ALIGNMENT: u32 = 32;

type Server = CraneLiftServer<DynamicMemoryManagement<CraneLiftStorage>>;

impl Runtime for CraneLiftRuntime {
    type Compiler = CraneLiftCompiler;
    type Server = CraneLiftServer<DynamicMemoryManagement<CraneLiftStorage>>;

    type Channel = MutexComputeChannel<CraneLiftServer<DynamicMemoryManagement<CraneLiftStorage>>>;
    type Device = CraneLiftDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        todo!()
    }

    fn name() -> &'static str {
        "craneLift"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn extension() -> &'static str {
        "wasm"
    }

    // how do I tweak this based on the device?
    fn supported_line_sizes() -> &'static [u8] {
        &[16, 8, 4, 2, 1]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        todo!()
    }
}

fn register_wmma_features(features: &mut FeatureSet, arch: u32) {
    let wmma_minimum_version = 70;
    let mut wmma = false;

    if arch >= wmma_minimum_version {
        wmma = true;
    }

    if wmma {
        // Types fully supported.
        for (a, b, c) in [
            (
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
            ),
            (
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F16),
                Elem::Float(FloatKind::F32),
            ),
            (
                Elem::Float(FloatKind::BF16),
                Elem::Float(FloatKind::BF16),
                Elem::Float(FloatKind::F32),
            ),
        ] {
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 16,
                k: 16,
                n: 16,
            });
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 32,
                k: 8,
                n: 16,
            });
            features.register(Feature::Cmma {
                a,
                b,
                c,
                m: 8,
                k: 32,
                n: 16,
            });
        }
    }
}
