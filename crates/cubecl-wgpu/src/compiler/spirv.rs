use ash::vk::EXT_ROBUSTNESS2_NAME;
use cubecl_common::ExecutionMode;
use cubecl_core::{prelude::CompiledKernel, server::ComputeServer};
use wgpu::hal::{self, vulkan};

use crate::{DynCompiler, WgpuServer};

use super::base::WgpuCompiler;

pub use cubecl_spirv::{GLCompute, SpirvCompiler};

pub type VkSpirvCompiler = SpirvCompiler<GLCompute>;

impl WgpuCompiler for SpirvCompiler<GLCompute> {
    fn compile(
        dyn_comp: &mut DynCompiler,
        server: &mut WgpuServer,
        kernel: <WgpuServer as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> CompiledKernel<DynCompiler> {
        // `wgpu` currently always enables `robustness2` on Vulkan if available, so default to
        // unchecked execution if robustness is enabled and let Vulkan handle it
        let mode = if is_robust(&server.device) {
            ExecutionMode::Unchecked
        } else {
            mode
        };
        log::debug!("Compiling {}", kernel.name());
        let compiled = kernel.compile(dyn_comp, &server.compilation_options, mode);
        #[cfg(feature = "spirv-dump")]
        dump_spirv(&compiled, kernel.name(), kernel.id());
        compiled
    }
}

fn is_robust(device: &wgpu::Device) -> bool {
    fn is_robust(device: &vulkan::Device) -> bool {
        device
            .enabled_device_extensions()
            .contains(&EXT_ROBUSTNESS2_NAME)
    }
    unsafe {
        device.as_hal::<hal::api::Vulkan, _, _>(|device| device.map(is_robust).unwrap_or(false))
    }
}

#[cfg(feature = "spirv-dump")]
fn dump_spirv(compiled: &CompiledKernel<DynCompiler>, name: &str, id: cubecl_core::KernelId) {
    use std::{
        fs,
        hash::{DefaultHasher, Hash, Hasher},
    };

    if let Ok(dir) = std::env::var("CUBECL_DEBUG_SPIRV") {
        if let Some(repr) = compiled.repr.as_ref().and_then(|repr| repr.as_spirv()) {
            let name = name
                .split("<")
                .take_while(|it| !it.ends_with("Runtime"))
                .map(|it| it.split(">").next().unwrap())
                .map(|it| it.split("::").last().unwrap())
                .collect::<Vec<_>>()
                .join("_");
            let mut hash = DefaultHasher::new();
            id.hash(&mut hash);
            let id = hash.finish();
            let name = sanitize_filename::sanitize_with_options(
                format!("{name}_{id:#x}"),
                sanitize_filename::Options {
                    replacement: "_",
                    ..Default::default()
                },
            );
            let kernel = repr.assemble().into_iter();
            let kernel = kernel.flat_map(|it| it.to_le_bytes()).collect::<Vec<_>>();
            fs::write(format!("{dir}/{name}.spv"), kernel).unwrap();
            fs::write(
                format!("{dir}/{name}.ir.txt"),
                format!("{}", repr.optimizer),
            )
            .unwrap();
        }
    }
}
