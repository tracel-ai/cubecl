use cubecl_core::{
    client::ComputeClient,
    prelude::{ArrayArg, TensorArg},
    server, Compiler, ExecutionMode, Kernel, Runtime,
};
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

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

pub fn compile(kernel: impl Kernel) -> String {
    <<WgpuRuntime as Runtime>::Compiler as Compiler>::compile(
        kernel.define(),
        ExecutionMode::Checked,
    )
    .to_string()
}
