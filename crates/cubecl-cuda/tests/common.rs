use std::{io::Write, process::Command};

use cubecl_core::{
    client::ComputeClient,
    prelude::{ArrayArg, TensorArg},
    server, Compiler, ExecutionMode, Kernel, Runtime,
};
use cubecl_cuda::{CudaDevice, CudaRuntime};

type Client = ComputeClient<<CudaRuntime as Runtime>::Server, <CudaRuntime as Runtime>::Channel>;
type Handle = server::Handle;

pub fn client() -> Client {
    let device = CudaDevice::new(0);
    CudaRuntime::client(&device)
}

#[allow(unused)]
pub fn handle(client: &Client) -> Handle {
    client.empty(1)
}

#[allow(unused)]
pub fn tensor(tensor: &Handle) -> TensorArg<'_, CudaRuntime> {
    unsafe { TensorArg::from_raw_parts(tensor, &[1], &[1], 1) }
}

#[allow(unused)]
pub fn tensor_vec(tensor: &Handle, vec: u8) -> TensorArg<'_, CudaRuntime> {
    unsafe { TensorArg::from_raw_parts(tensor, &[1], &[1], vec) }
}

#[allow(unused)]
pub fn array(tensor: &Handle) -> ArrayArg<'_, CudaRuntime> {
    unsafe { ArrayArg::from_raw_parts(tensor, 1, 1) }
}

pub fn compile(kernel: impl Kernel) -> String {
    let kernel = <<CudaRuntime as Runtime>::Compiler as Compiler>::compile(
        kernel.define(),
        ExecutionMode::Checked,
    )
    .to_string();
    format_cpp_code(&kernel).unwrap()
}

/// Format C++ code, useful when debugging.
fn format_cpp_code(code: &str) -> Result<String, std::io::Error> {
    let mut child = Command::new("clang-format")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()?;

    {
        let stdin = child.stdin.as_mut().expect("Failed to open stdin");
        stdin.write_all(code.as_bytes())?;
    }

    let output = child.wait_with_output()?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "clang-format failed",
        ))
    }
}
