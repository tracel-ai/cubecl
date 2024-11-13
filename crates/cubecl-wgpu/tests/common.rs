use std::num::NonZero;

use cubecl_core::{
    prelude::{ArrayCompilationArg, TensorCompilationArg},
    Compiler, CubeDim, ExecutionMode, Kernel, KernelSettings, Runtime,
};
use cubecl_wgpu::{WgpuRuntime, WgslCompiler};

pub type TestRuntime = WgpuRuntime<WgslCompiler>;

pub fn settings(dim_x: u32, dim_y: u32) -> KernelSettings {
    KernelSettings::default().cube_dim(CubeDim::new(dim_x, dim_y, 1))
}

#[allow(unused)]
pub fn tensor() -> TensorCompilationArg {
    TensorCompilationArg {
        inplace: None,
        vectorisation: NonZero::new(1),
    }
}

#[allow(unused)]
pub fn tensor_vec(vectorization: u8) -> TensorCompilationArg {
    TensorCompilationArg {
        inplace: None,
        vectorisation: NonZero::new(vectorization),
    }
}

#[allow(unused)]
pub fn array() -> ArrayCompilationArg {
    ArrayCompilationArg {
        inplace: None,
        vectorisation: NonZero::new(1),
    }
}

pub fn compile(kernel: impl Kernel) -> String {
    <<TestRuntime as Runtime>::Compiler as Compiler>::compile(
        kernel.define(),
        ExecutionMode::Checked,
    )
    .to_string()
}

#[macro_export]
macro_rules! load_kernel_string {
    ($file:expr) => {
        include_str!($file)
            .replace("\r\n", "\n")
            .trim_end()
            .to_string()
    };
}
