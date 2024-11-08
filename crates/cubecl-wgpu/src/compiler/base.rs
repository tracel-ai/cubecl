use cubecl_core::{
    prelude::CompiledKernel, server::ComputeServer, Compiler, ExecutionMode, Feature,
};
use cubecl_runtime::DeviceProperties;
use wgpu::{Adapter, ComputePipeline, Device, Queue};

use crate::{Pdrc, WgpuServer, WgpuServerInner};

pub trait WgpuCompiler: Compiler {
    fn compile(
        server: &mut WgpuServerInner<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> CompiledKernel<Self>;

    fn create_pipeline(
        server: &mut WgpuServerInner<Self>,
        kernel: CompiledKernel<Self>,
        mode: ExecutionMode,
    ) -> Pdrc<ComputePipeline>;

    #[allow(async_fn_in_trait)]
    async fn request_device(adapter: &Adapter) -> (Device, Queue);
    fn register_features(adapter: &Adapter, device: &Device, props: &mut DeviceProperties<Feature>);
}
