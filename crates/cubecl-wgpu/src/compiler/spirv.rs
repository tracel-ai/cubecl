use std::{borrow::Cow, sync::Arc};

use ash::{
    khr::cooperative_matrix,
    vk::{
        ComponentTypeKHR, DeviceCreateInfo, DeviceQueueCreateInfo,
        PhysicalDevice16BitStorageFeatures, PhysicalDeviceCooperativeMatrixFeaturesKHR,
        PhysicalDeviceShaderFloat16Int8Features, PhysicalDeviceVulkanMemoryModelFeatures, ScopeKHR,
        EXT_ROBUSTNESS2_NAME, KHR_COOPERATIVE_MATRIX_NAME,
    },
};
use cubecl_core::{
    channel::MutexComputeChannel,
    client::ComputeClient,
    future,
    ir::{Elem, FloatKind, IntKind},
    prelude::CompiledKernel,
    server::ComputeServer,
    ExecutionMode, Feature, Runtime,
};
use cubecl_runtime::{ComputeRuntime, DeviceProperties};
use wgpu::{
    hal::{self, vulkan},
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    ComputePipeline, DeviceDescriptor, Features, Limits, PipelineLayoutDescriptor,
    ShaderModuleDescriptorSpirV, ShaderStages,
};

use crate::{
    create_client, create_wgpu_setup, RuntimeOptions, Vulkan, WgpuDevice, WgpuRuntime, WgpuServer,
};

use super::base::WgpuCompiler;

pub use cubecl_spirv::{GLCompute, SpirvCompiler};
pub type VkSpirvCompiler = SpirvCompiler<GLCompute>;

type Server = WgpuServer<SpirvCompiler<GLCompute>>;

/// The compute instance is shared across all [wgpu runtimes](WgpuRuntime).
static RUNTIME: ComputeRuntime<WgpuDevice, Server, MutexComputeChannel<Server>> =
    ComputeRuntime::new();

impl WgpuCompiler for SpirvCompiler<GLCompute> {
    fn create_pipeline(
        server: &mut WgpuServer<Self>,
        kernel: CompiledKernel<Self>,
        _mode: ExecutionMode,
    ) -> Arc<ComputePipeline> {
        let repr = kernel
            .repr
            .expect("Need compiled repr to assemble to spirv");
        let spirv = repr.assemble();

        let num_bindings = repr.num_bindings as u32;
        let bindings = (0..num_bindings)
            .map(|i| BindGroupLayoutEntry {
                binding: i,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let layout = server
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &bindings,
            });
        let layout = server
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

        let module = unsafe {
            server
                .device
                .create_shader_module_spirv(&ShaderModuleDescriptorSpirV {
                    label: None,
                    source: Cow::Borrowed(&spirv),
                })
        };

        Arc::new(
            server
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&layout),
                    module: &module,
                    entry_point: "main",
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
        let compiled = kernel.compile(mode);
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
    ) {
        register_types(props);
        let cmma = unsafe {
            adapter.as_hal::<hal::api::Vulkan, _, _>(|adapter| {
                let adapter = adapter.expect("Can only use SPIR-V with Vulkan");
                let pd = adapter.raw_physical_device();
                let ash = adapter.shared_instance();
                let cmma = cooperative_matrix::Instance::new(ash.entry(), ash.raw_instance());
                let properties = cmma
                    .get_physical_device_cooperative_matrix_properties(pd)
                    .unwrap();
                properties
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
                    .collect::<Vec<_>>()
            })
        };
        for size in cmma {
            props.register_feature(size);
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

    let has_cmma = adapter
        .physical_device_capabilities()
        .supports_extension(KHR_COOPERATIVE_MATRIX_NAME);
    let mut device_extensions = adapter.required_device_extensions(features);
    let mut cmma = None;
    let mut mem_model = PhysicalDeviceVulkanMemoryModelFeatures::default()
        .vulkan_memory_model(true)
        .vulkan_memory_model_device_scope(true);
    let mut f16_i8 = PhysicalDeviceShaderFloat16Int8Features::default()
        .shader_float16(true)
        .shader_int8(true);
    let mut buf_16 =
        PhysicalDevice16BitStorageFeatures::default().storage_buffer16_bit_access(true);

    if has_cmma {
        device_extensions.push(KHR_COOPERATIVE_MATRIX_NAME);
        cmma = Some(PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true))
    }

    let mut phys_features = adapter.physical_device_features(&device_extensions, features);
    let ash = adapter.shared_instance();

    let supported_feat = unsafe {
        ash.raw_instance()
            .get_physical_device_features(adapter.raw_physical_device())
    };

    let family_index = 0; //TODO
    let family_info = DeviceQueueCreateInfo::default()
        .queue_family_index(family_index)
        .queue_priorities(&[1.0]);
    let family_infos = [family_info];

    let str_pointers = device_extensions
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
    info = info.push_next(&mut mem_model);
    info = info.push_next(&mut f16_i8);
    info = info.push_next(&mut buf_16);

    if let Some(cmma) = &mut cmma {
        info = info.push_next(cmma);
    }

    let vk_device = unsafe {
        ash.raw_instance()
            .create_device(adapter.raw_physical_device(), &info, None)
            .expect("Failed to create Vulkan device")
    };

    let device = unsafe {
        adapter
            .device_from_raw(
                vk_device,
                true,
                &device_extensions,
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

fn register_types(props: &mut DeviceProperties<Feature>) {
    use cubecl_core::ir::{Elem, FloatKind, IntKind};

    let supported_types = [
        Elem::UInt,
        Elem::Int(IntKind::I32),
        Elem::Int(IntKind::I64),
        Elem::AtomicInt(IntKind::I32),
        Elem::AtomicInt(IntKind::I64),
        Elem::AtomicUInt,
        Elem::Float(FloatKind::F16),
        Elem::Float(FloatKind::F32),
        Elem::Float(FloatKind::F64),
        Elem::Bool,
    ];

    for ty in supported_types {
        props.register_feature(Feature::Type(ty));
    }
}

fn conv_type(vk_ty: ComponentTypeKHR) -> Option<Elem> {
    let ty = match vk_ty {
        ComponentTypeKHR::FLOAT16 => Elem::Float(FloatKind::F16),
        ComponentTypeKHR::FLOAT32 => Elem::Float(FloatKind::F32),
        ComponentTypeKHR::FLOAT64 => Elem::Float(FloatKind::F64),
        ComponentTypeKHR::SINT32 => Elem::Int(IntKind::I32),
        ComponentTypeKHR::SINT64 => Elem::Int(IntKind::I64),
        ComponentTypeKHR::UINT32 => Elem::UInt,
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
            let (adapter, device_wgpu, queue) =
                future::block_on(create_wgpu_setup::<Vulkan, VkSpirvCompiler>(device));
            create_client(adapter, device_wgpu, queue, RuntimeOptions::default())
        })
    }

    fn name() -> &'static str {
        "wgpu<spirv>"
    }

    fn supported_line_sizes() -> &'static [u8] {
        &[4, 2]
    }
}

pub fn init_existing_device(
    adapter: Arc<wgpu::Adapter>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    options: RuntimeOptions,
) -> WgpuDevice {
    let device_id = WgpuDevice::Existing(device.as_ref().global_id());
    let client = create_client(adapter, device, queue, options);
    RUNTIME.register(&device_id, client);
    device_id
}

/// Initialize a client on the given device with the given options. This function is useful to configure the runtime options
/// or to pick a different graphics API. On wasm, it is necessary to use [`init_async`] instead.
pub fn init_sync(device: &WgpuDevice, options: RuntimeOptions) {
    future::block_on(init_async(device, options));
}

/// Like [`init_sync`], but async, necessary for wasm.
pub async fn init_async(device: &WgpuDevice, options: RuntimeOptions) {
    let (adapter, device_wgpu, queue) =
        create_wgpu_setup::<Vulkan, SpirvCompiler<GLCompute>>(device).await;
    let client = create_client(adapter, device_wgpu, queue, options);
    RUNTIME.register(device, client)
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
