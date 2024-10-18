use cubecl_core::{
    client::ComputeClient,
    prelude::{ArrayArg, TensorArg},
    server::Handle,
    Compiler, ExecutionMode, Kernel, Runtime,
};
use cubecl_wgpu::{WgpuDevice, WgpuRuntime, WgslCompiler};

pub type TestRuntime = WgpuRuntime<WgslCompiler>;

type Client = ComputeClient<<TestRuntime as Runtime>::Server, <TestRuntime as Runtime>::Channel>;

pub fn client() -> Client {
    let device = WgpuDevice::default();
    TestRuntime::client(&device)
}

#[allow(unused)]
pub fn handle(client: &Client) -> Handle {
    client.empty(1)
}

#[allow(unused)]
pub fn tensor(tensor: &Handle) -> TensorArg<'_, TestRuntime> {
    unsafe { TensorArg::from_raw_parts(tensor, &[1], &[1], 1) }
}

#[allow(unused)]
pub fn tensor_vec(tensor: &Handle, vectorization: u8) -> TensorArg<'_, TestRuntime> {
    unsafe { TensorArg::from_raw_parts(tensor, &[1], &[1], vectorization) }
}

#[allow(unused)]
pub fn array(tensor: &Handle) -> ArrayArg<'_, TestRuntime> {
    unsafe { ArrayArg::from_raw_parts(tensor, 1, 1) }
}

pub fn compile(kernel: impl Kernel) -> String {
    <<TestRuntime as Runtime>::Compiler as Compiler>::compile(
        kernel.define(),
        ExecutionMode::Checked,
    )
    .to_string()
}
