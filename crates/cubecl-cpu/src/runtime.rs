use cubecl_core::{DeviceId, Runtime, channel::MutexComputeChannel};
use cubecl_runtime::ComputeRuntime;

use crate::{compiler::MLIRCompiler, compute::server::CpuServer, device::CpuDevice};

#[derive(Debug)]
pub struct CpuRuntime;

type Server = CpuServer;
type Channel = MutexComputeChannel<Server>;

static RUNTIME: ComputeRuntime<CpuDevice, Server, Channel> = ComputeRuntime::new();

pub type CpuCompiler = MLIRCompiler;

impl Runtime for CpuRuntime {
    type Compiler = CpuCompiler;
    type Server = CpuServer;

    type Channel = Channel;
    type Device = CpuDevice;

    fn device_id(device: &Self::Device) -> cubecl_core::DeviceId {
        DeviceId::new(0, 0)
    }

    fn client(
        device: &Self::Device,
    ) -> cubecl_core::prelude::ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            todo!();
        })
    }

    fn name(
        client: &cubecl_core::prelude::ComputeClient<Self::Server, Self::Channel>,
    ) -> &'static str {
        "cpu"
    }

    // TODO find line size corresponding to the architecture
    // for AVX512, AVX2 et Neon for the moment
    fn supported_line_sizes() -> &'static [u8] {
        &[8, 1, 1, 1]
    }

    fn max_cube_count() -> (u32, u32, u32) {
        todo!()
    }

    fn can_read_tensor(shape: &[usize], strides: &[usize]) -> bool {
        todo!()
    }
}
