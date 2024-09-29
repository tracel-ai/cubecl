use cubecl_core::{
    client::ComputeClient,
    prelude::{ArrayArg, TensorArg},
    server, Compiler, ExecutionMode, Kernel, Runtime,
};
use cubecl_spirv::{GLCompute, SpirvCompiler};
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
use rspirv::{binary::Assemble, dr::Module};

type Client = ComputeClient<<WgpuRuntime as Runtime>::Server, <WgpuRuntime as Runtime>::Channel>;
type Handle = server::Handle<<WgpuRuntime as Runtime>::Server>;

pub fn client() -> Client {
    let device = WgpuDevice::default();
    WgpuRuntime::client(&device)
}

#[allow(unused)]
pub fn handle(client: &Client) -> Handle {
    client.empty(1)
}

#[allow(unused)]
pub fn tensor(tensor: &Handle) -> TensorArg<'_, WgpuRuntime> {
    unsafe { TensorArg::from_raw_parts(tensor, &[1], &[1], 1) }
}

#[allow(unused)]
pub fn tensor_vec(tensor: &Handle, vectorization: u8) -> TensorArg<'_, WgpuRuntime> {
    unsafe { TensorArg::from_raw_parts(tensor, &[1], &[1], vectorization) }
}

#[allow(unused)]
pub fn array(tensor: &Handle) -> ArrayArg<'_, WgpuRuntime> {
    unsafe { ArrayArg::from_raw_parts(tensor, 1, 1) }
}

#[allow(unused)]
pub fn array_vec(tensor: &Handle, factor: u8) -> ArrayArg<'_, WgpuRuntime> {
    unsafe { ArrayArg::from_raw_parts(tensor, 1, factor) }
}

#[allow(unused)]
pub fn compile(kernel: impl Kernel) -> Module {
    SpirvCompiler::<GLCompute>::compile(kernel.define(), ExecutionMode::Checked).module
}

#[allow(unused)]
pub fn compile_unchecked(kernel: impl Kernel) -> Module {
    SpirvCompiler::<GLCompute>::compile(kernel.define(), ExecutionMode::Unchecked).module
}

pub fn to_bytes(module: Module) -> Vec<u8> {
    module
        .assemble()
        .into_iter()
        .flat_map(|it| it.to_le_bytes())
        .collect()
}
