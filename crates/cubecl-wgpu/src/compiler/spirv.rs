use std::{borrow::Cow, sync::Arc};

use ash::{
    khr::cooperative_matrix,
    vk::{
        ComponentTypeKHR, DeviceCreateInfo, DeviceQueueCreateInfo, ScopeKHR, EXT_ROBUSTNESS2_NAME,
        TRUE,
    },
};
use cubecl_core::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    future,
    ir::{Elem, FloatKind, IntKind, UIntKind},
    prelude::CompiledKernel,
    server::ComputeServer,
    AtomicFeature, ExecutionMode, Feature, Runtime,
};
use cubecl_runtime::{ComputeRuntime, DeviceProperties};
use cubecl_spirv::CompilationOptions;
use features::ExtendedFeatures;
use wgpu::{
    hal::{
        self,
        vulkan::{self, InstanceShared},
    },
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, DeviceDescriptor, Features, Limits, PipelineLayoutDescriptor,
    ShaderModuleDescriptorSpirV, ShaderStages,
};

use crate::{
    create_client_on_setup, create_setup_for_device, RuntimeOptions, Vulkan, WgpuDevice,
    WgpuRuntime, WgpuServer,
};

use super::base::WgpuCompiler;

pub use cubecl_spirv::{GLCompute, SpirvCompiler};

mod features;

pub type VkSpirvCompiler = SpirvCompiler<GLCompute>;

type Server = WgpuServer<SpirvCompiler<GLCompute>>;

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

impl WgpuCompiler for SpirvCompiler<GLCompute> {
    fn create_pipeline(
        server: &mut WgpuServer<Self>,
        kernel: CompiledKernel<Self>,
    ) -> Arc<ComputePipeline> {
        let (module, layout) = kernel
            .repr
            .map(|repr| {
                let bindings = repr
                    .bindings
                    .iter()
                    .enumerate()
                    .map(|(i, _binding)| BindGroupLayoutEntry {
                        binding: i as u32,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            #[cfg(not(exclusive_memory_only))]
                            ty: BufferBindingType::Storage { read_only: false },
                            #[cfg(exclusive_memory_only)]
                            ty: BufferBindingType::Storage {
                                read_only: matches!(
                                    _binding.visibility,
                                    cubecl_core::ir::Visibility::Read
                                ),
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>();
                let layout = server
                    .device
                    .create_bind_group_layout(&BindGroupLayoutDescriptor {
                        label: Some(&kernel.entrypoint_name),
                        entries: &bindings,
                    });
                let layout = server
                    .device
                    .create_pipeline_layout(&PipelineLayoutDescriptor {
                        label: Some(&kernel.entrypoint_name),
                        bind_group_layouts: &[&layout],
                        push_constant_ranges: &[],
                    });

                let spirv = repr.assemble();

                let module = unsafe {
                    server
                        .device
                        .create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
                            label: Some(&kernel.entrypoint_name),
                            source: Cow::Borrowed(&spirv),
                        })
                };
                (module, Some(layout))
            })
            .unwrap_or_else(|| {
                let source = &kernel.source;
                // Cube always in principle uses unchecked modules. Certain operations like
                // indexing are instead checked by cube. The WebGPU specification only makes
                // incredibly loose guarantees that Cube can't rely on. Additionally, kernels
                // can opt in/out per operation whether checks should be performed which can be faster.
                //
                // SAFETY: Cube guarantees OOB safety when launching in checked mode. Launching in unchecked mode
                // is only available through the use of unsafe code.
                let module = unsafe {
                    server
                        .device
                        .create_shader_module_unchecked(wgpu::ShaderModuleDescriptor {
                            label: Some(&kernel.entrypoint_name),
                            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
                        })
                };
                (module, None)
            });

        Arc::new(
            server
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&kernel.entrypoint_name),
                    layout: layout.as_ref(),
                    module: &module,
                    entry_point: Some(&kernel.entrypoint_name),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: None,
                }),
        )
    }

    fn compile(
        server: &mut WgpuServer<Self>,
        kernel: <WgpuServer<Self> as ComputeServer>::Kernel,
        mode: ExecutionMode,
    ) -> CompiledKernel<Self> {
        // `wgpu` currently always enables `robustness2` on Vulkan if available, so default to
        // unchecked execution if robustness is enabled and let Vulkan handle it
        let mode = if is_robust(&server.device) {
            ExecutionMode::Unchecked
        } else {
            mode
        };
        log::debug!("Compiling {}", kernel.name());
        let compiled = kernel.compile(&server.compilation_options, mode);
        #[cfg(feature = "spirv-dump")]
        dump_spirv(&compiled, kernel.name(), kernel.id());
        compiled
    }

    async fn request_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
        let limits = adapter.limits();
        let features = adapter.features();
        unsafe {
            adapter.as_hal::<hal::api::Vulkan, _, _>(|hal_adapter| {
                request_device(adapter, hal_adapter.unwrap(), features, limits)
            })
        }
    }

    fn register_features(
        adapter: &wgpu::Adapter,
        _device: &wgpu::Device,
        props: &mut cubecl_runtime::DeviceProperties<cubecl_core::Feature>,
        comp_options: &mut Self::CompilationOptions,
    ) {
        let features = adapter.features();
        unsafe {
            adapter.as_hal::<hal::api::Vulkan, _, _>(|hal_adapter| {
                if let Some(adapter) = hal_adapter {
                    register_features(adapter, props, features, comp_options);
                }
            })
        }
    }
}

/// Request device with required features, plus CMMA if available.
fn request_device(
    wgpu_adapter: &wgpu::Adapter,
    adapter: &vulkan::Adapter,
    mut features: Features,
    limits: Limits,
) -> (wgpu::Device, wgpu::Queue) {
    // This registers only f16 but not u8/i8, so remove so we can manually add them
    features.remove(Features::SHADER_F16);

    let ash = adapter.shared_instance();
    let mut extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);
    let extensions = extended_feat.extensions.clone();
    let mut phys_features = adapter.physical_device_features(&extended_feat.extensions, features);

    let supported_feat = unsafe {
        ash.raw_instance()
            .get_physical_device_features(adapter.raw_physical_device())
    };

    let family_index = 0; //TODO
    let family_info = DeviceQueueCreateInfo::default()
        .queue_family_index(family_index)
        .queue_priorities(&[1.0]);
    let family_infos = [family_info];

    let str_pointers = extended_feat
        .extensions
        .iter()
        .map(|&s| {
            // Safe because `device_extensions` entries have static lifetime.
            s.as_ptr()
        })
        .collect::<Vec<_>>();

    let pre_info = DeviceCreateInfo::default()
        .queue_create_infos(&family_infos)
        .enabled_extension_names(&str_pointers);
    let mut info = phys_features.add_to_device_create(pre_info);
    info = info.enabled_features(&supported_feat);
    info = extended_feat.add_to_device_create(info);

    let vk_device = unsafe {
        ash.raw_instance()
            .create_device(adapter.raw_physical_device(), &info, None)
            .expect("Failed to create Vulkan device")
    };

    let device = unsafe {
        adapter
            .device_from_raw(
                vk_device,
                None,
                &extensions,
                features,
                &wgpu::MemoryHints::MemoryUsage,
                family_info.queue_family_index,
                0,
            )
            .expect("Failed to create HAL device")
    };

    let descriptor = DeviceDescriptor {
        label: None,
        required_features: features,
        required_limits: limits,
        // The default is MemoryHints::Performance, which tries to do some bigger
        // block allocations. However, we already batch allocations, so we
        // can use MemoryHints::MemoryUsage to lower memory usage.
        memory_hints: wgpu::MemoryHints::MemoryUsage,
    };

    unsafe {
        wgpu_adapter
            .create_device_from_hal(device, &descriptor, None)
            .expect("Failed to create wgpu device")
    }
}

/// Request device's supported features
fn register_features(
    adapter: &vulkan::Adapter,
    props: &mut cubecl_runtime::DeviceProperties<cubecl_core::Feature>,
    features: Features,
    comp_options: &mut CompilationOptions,
) {
    let ash = adapter.shared_instance();
    let extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);

    log::debug!("Supported Vulkan features: {extended_feat:#?}");

    register_types(props, &extended_feat);

    if let Some(atomic_float) = &extended_feat.atomic_float {
        if atomic_float.shader_buffer_float32_atomics == TRUE {
            props.register_feature(Feature::AtomicFloat(AtomicFeature::LoadStore));
        }
        if atomic_float.shader_buffer_float32_atomic_add == TRUE {
            props.register_feature(Feature::AtomicFloat(AtomicFeature::Add));
        }
    }
    if let Some(atomic_float2) = &extended_feat.atomic_float2 {
        if atomic_float2.shader_buffer_float32_atomic_min_max == TRUE {
            props.register_feature(Feature::AtomicFloat(AtomicFeature::MinMax));
        }
    }

    if let Some(float_controls2) = &extended_feat.float_controls2 {
        if float_controls2.shader_float_controls2 == TRUE {
            comp_options.supports_fp_fast_math = true;
        }
    }

    if extended_feat.cmma.is_some() {
        register_cmma(ash, adapter, props);
    }
}

fn register_types(props: &mut DeviceProperties<Feature>, ext_feat: &ExtendedFeatures<'_>) {
    use cubecl_core::ir::{Elem, FloatKind, IntKind};

    let mut register = |elem| {
        props.register_feature(Feature::Type(elem));
    };

    let supported_types = [
        Elem::UInt(UIntKind::U16),
        Elem::UInt(UIntKind::U32),
        Elem::UInt(UIntKind::U64),
        Elem::Int(IntKind::I16),
        Elem::Int(IntKind::I32),
        Elem::Int(IntKind::I64),
        Elem::AtomicInt(IntKind::I32),
        Elem::AtomicInt(IntKind::I64),
        Elem::AtomicUInt(UIntKind::U32),
        Elem::AtomicUInt(UIntKind::U64),
        Elem::Float(FloatKind::F32),
        //Elem::Float(FloatKind::F64),
        Elem::Bool,
    ];

    for ty in supported_types {
        register(ty);
    }

    if ext_feat.float16_int8.shader_float16 == TRUE {
        register(Elem::Float(FloatKind::F16));
    }
    if ext_feat.float16_int8.shader_int8 == TRUE {
        register(Elem::Int(IntKind::I8));
        register(Elem::UInt(UIntKind::U8));
    }

    if let Some(atomic_float) = ext_feat.atomic_float {
        if atomic_float.shader_buffer_float32_atomics == TRUE {
            register(Elem::AtomicFloat(FloatKind::F32));
            register(Elem::AtomicFloat(FloatKind::F64));
        }
    }
    if let Some(atomic_float) = ext_feat.atomic_float2 {
        if atomic_float.shader_buffer_float16_atomics == TRUE {
            register(Elem::AtomicFloat(FloatKind::F16));
        }
    }
}

fn register_cmma(
    ash: &InstanceShared,
    adapter: &vulkan::Adapter,
    props: &mut DeviceProperties<Feature>,
) {
    let cmma = cooperative_matrix::Instance::new(ash.entry(), ash.raw_instance());
    let properties = unsafe {
        cmma.get_physical_device_cooperative_matrix_properties(adapter.raw_physical_device())
            .unwrap()
    };
    let sizes = properties
        .into_iter()
        .filter(|it| {
            it.saturating_accumulation == 0
                && it.result_type == it.c_type
                && it.scope == ScopeKHR::SUBGROUP
        })
        .filter_map(|it| {
            Some(Feature::Cmma {
                a: conv_type(it.a_type)?,
                b: conv_type(it.b_type)?,
                c: conv_type(it.c_type)?,
                m: it.m_size as u8,
                k: it.k_size as u8,
                n: it.n_size as u8,
            })
        })
        .collect::<Vec<_>>();
    log::debug!("Supported CMMA sizes: {sizes:#?}");
    for size in sizes {
        props.register_feature(size);
    }
}

fn conv_type(vk_ty: ComponentTypeKHR) -> Option<Elem> {
    let ty = match vk_ty {
        ComponentTypeKHR::FLOAT16 => Elem::Float(FloatKind::F16),
        ComponentTypeKHR::FLOAT32 => Elem::Float(FloatKind::F32),
        ComponentTypeKHR::FLOAT64 => Elem::Float(FloatKind::F64),
        ComponentTypeKHR::SINT8 => Elem::Int(IntKind::I8),
        ComponentTypeKHR::SINT16 => Elem::Int(IntKind::I16),
        ComponentTypeKHR::SINT32 => Elem::Int(IntKind::I32),
        ComponentTypeKHR::SINT64 => Elem::Int(IntKind::I64),
        ComponentTypeKHR::UINT8 => Elem::UInt(UIntKind::U8),
        ComponentTypeKHR::UINT16 => Elem::UInt(UIntKind::U16),
        ComponentTypeKHR::UINT32 => Elem::UInt(UIntKind::U32),
        ComponentTypeKHR::UINT64 => Elem::UInt(UIntKind::U64),
        _ => None?,
    };
    Some(ty)
}

fn is_robust(device: &wgpu::Device) -> bool {
    fn is_robust(device: &vulkan::Device) -> bool {
        device
            .enabled_device_extensions()
            .contains(&EXT_ROBUSTNESS2_NAME)
    }
    unsafe {
        device
            .as_hal::<hal::api::Vulkan, _, _>(|device| device.map(is_robust).unwrap_or(false))
            .unwrap_or(false)
    }
}

impl Runtime for WgpuRuntime<VkSpirvCompiler> {
    type Compiler = VkSpirvCompiler;
    type Server = WgpuServer<VkSpirvCompiler>;

    type Channel = MutexComputeChannel<WgpuServer<VkSpirvCompiler>>;
    type Device = WgpuDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self::Server, Self::Channel> {
        RUNTIME.client(device, move || {
            let setup =
                future::block_on(create_setup_for_device::<Vulkan, VkSpirvCompiler>(device));
            create_client_on_setup(setup, RuntimeOptions::default())
        })
    }

    fn name() -> &'static str {
        "wgpu<spirv>"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[4, 2]
    }

    fn extension() -> &'static str {
        "spv"
    }

    fn max_cube_count() -> (u32, u32, u32) {
        let max_dim = u16::MAX as u32;
        (max_dim, max_dim, max_dim)
    }
}

#[cfg(feature = "spirv-dump")]
fn dump_spirv(compiled: &CompiledKernel<VkSpirvCompiler>, name: &str, id: cubecl_core::KernelId) {
    use std::{
        fs,
        hash::{DefaultHasher, Hash, Hasher},
    };

    if let Ok(dir) = std::env::var("CUBECL_DEBUG_SPIRV") {
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
        let repr = compiled.repr.as_ref().unwrap();
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
