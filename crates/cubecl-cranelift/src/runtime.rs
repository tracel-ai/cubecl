use std::mem::MaybeUninit;

use cubecl_core::Runtime;
use cubecl_runtime::{channel::MutexComputeChannel, client::ComputeClient, ComputeRuntime};

use crate::{
    compiler::FunctionCompiler,
    compute::{CraneliftContext, CraneliftServer, CraneliftStorage},
    device::CraneLiftDevice,
};

#[derive(Debug)]
pub struct CraneLiftRuntime;

static RUNTIME: ComputeRuntime<CraneLiftDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

const MEMORY_OFFSET_ALIGNMENT: u32 = 32;

type Server = CraneliftServer;

impl Runtime for CraneLiftRuntime {
    type Compiler = FunctionCompiler;
    type Server = CraneliftServer;

    type Channel = MutexComputeChannel<Server>;
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
