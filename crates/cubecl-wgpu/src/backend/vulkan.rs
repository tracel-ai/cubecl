use ash::vk::{API_VERSION_1_1, KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME, MemoryHeapFlags};
use cubecl_core::{
    ExecutionMode, MemoryConfiguration, WgpuCompilationOptions,
    ir::{AddressType, ElemType, FloatKind, IntKind, UIntKind},
    prelude::{CompiledKernel, Visibility},
    server::{ComputeServer, KernelArguments},
};
use cubecl_ir::{DeviceProperties, Type, features::*};
use cubecl_runtime::compiler::CompilationError;
use cubecl_spirv::{GLCompute, SpirvCompiler, SpirvKernel};
use features::ExtendedFeatures;
use tracel_ash::{
    khr::cooperative_matrix,
    vk::{ComponentTypeKHR, DeviceCreateInfo, DeviceQueueCreateInfo, ScopeKHR, TRUE},
};
use wgpu::{
    DeviceDescriptor, Features, Limits,
    hal::{
        self,
        vulkan::{self, InstanceShared},
    },
};

use crate::{AutoCompiler, WgpuServer};

mod features;

pub type VkSpirvCompiler = SpirvCompiler<GLCompute>;

pub fn bindings(
    repr: &SpirvKernel,
    bindings: &KernelArguments,
) -> (Vec<Visibility>, Option<Visibility>, bool) {
    let buffers: Vec<_> = repr.bindings.clone();
    let meta = (!bindings.info.data.is_empty()).then_some(Visibility::Read);
    (buffers, meta, repr.uniform_info)
}

pub async fn request_vulkan_device(adapter: &wgpu::Adapter) -> Option<(wgpu::Device, wgpu::Queue)> {
    let limits = adapter.limits();
    let features = adapter
        .features()
        .difference(Features::MAPPABLE_PRIMARY_BUFFERS);
    unsafe {
        let hal_adapter = adapter.as_hal::<hal::api::Vulkan>().unwrap();
        request_device(adapter, &hal_adapter, features, limits)
    }
}

pub fn register_vulkan_features(
    adapter: &wgpu::Adapter,
    props: &mut DeviceProperties,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) -> bool {
    let features = adapter.features();
    unsafe {
        if let Some(adapter) = adapter.as_hal::<hal::api::Vulkan>() {
            register_features(&adapter, props, features, comp_options, memory_config)
        } else {
            false
        }
    }
}

/// Request device with required features, plus CMMA if available.
fn request_device(
    wgpu_adapter: &wgpu::Adapter,
    adapter: &vulkan::Adapter,
    features: Features,
    mut limits: Limits,
) -> Option<(wgpu::Device, wgpu::Queue)> {
    let ash = adapter.shared_instance();

    // Can't even query for required features without `PhysicalDeviceFeatures2`
    if ash.instance_api_version() < API_VERSION_1_1
        && !ash
            .extensions()
            .contains(&KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME)
    {
        return None;
    }

    let mut extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);

    if !extended_feat.has_required_features() {
        return None;
    }

    let extensions = adapter.required_device_extensions(features);
    let mut phys_features = adapter.physical_device_features(&extensions, features);

    if let Some(index_64) = &extended_feat.index_64
        && index_64.shader64_bit_indexing == TRUE
    {
        limits.max_storage_buffer_binding_size = u64::MAX;
    }

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
    let mut info = phys_features.add_to_device_create(pre_info.into());
    info = info.enabled_features(&supported_feat);
    info = extended_feat.add_to_device_create(info.into()).into();

    let vk_device = unsafe {
        ash.raw_instance()
            .create_device(adapter.raw_physical_device(), &info, None)
            .expect("Failed to create Vulkan device")
    };

    // The default is MemoryHints::Performance, which tries to do some bigger
    // block allocations. However, we already batch allocations, so we
    // can use MemoryHints::MemoryUsage to lower memory usage.
    let memory_hints = wgpu::MemoryHints::MemoryUsage;
    let device = unsafe {
        adapter
            .device_from_raw(
                vk_device,
                None,
                &extensions,
                features,
                &limits,
                &memory_hints,
                family_info.queue_family_index,
                0,
            )
            .expect("Failed to create HAL device")
    };

    let descriptor = DeviceDescriptor {
        label: None,
        required_features: features,
        required_limits: limits,
        memory_hints,
        trace: wgpu::Trace::Off,
        // SAFETY: Enabling experimental passthrough shaders.
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    };

    unsafe {
        Some(
            wgpu_adapter
                .create_device_from_hal(device, &descriptor)
                .expect("Failed to create wgpu device"),
        )
    }
}

/// Request device's supported features
fn register_features(
    adapter: &vulkan::Adapter,
    props: &mut DeviceProperties,
    features: Features,
    comp_options: &mut WgpuCompilationOptions,
    memory_config: &MemoryConfiguration,
) -> bool {
    let ash = adapter.shared_instance();

    // Can't even query for required features without `PhysicalDeviceFeatures2`
    if ash.instance_api_version() < API_VERSION_1_1
        && !ash
            .extensions()
            .contains(&KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME)
    {
        return false;
    }

    let extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);

    if !extended_feat.has_required_features() {
        return false;
    }

    log::debug!("Supported Vulkan features: {extended_feat:#?}");

    register_types(props, &extended_feat);

    comp_options.supports_vulkan = true;
    comp_options.supports_u64 = extended_feat.core.shader_int64 == TRUE;
    comp_options.vulkan.max_spirv_version = extended_feat.max_spirv_version;

    props.features.plane.insert(Plane::Sync);

    if let Some(uniform_standard_layout) = extended_feat.uniform_standard_layout
        && uniform_standard_layout.uniform_buffer_standard_layout == TRUE
    {
        comp_options.vulkan.supports_uniform_standard_layout = true;
    }

    if let Some(uniform_unsized_array) = extended_feat.uniform_unsized_array
        && uniform_unsized_array.shader_uniform_buffer_unsized_array == TRUE
    {
        comp_options.vulkan.supports_uniform_unsized_array = true;
    }

    if let Some(float_controls2) = &extended_feat.float_controls2
        && float_controls2.shader_float_controls2 == TRUE
    {
        comp_options.vulkan.supports_fp_fast_math = true;
    }

    if let Some(wg_explicit_layout) = &extended_feat.wg_explicit_layout
        && wg_explicit_layout.workgroup_memory_explicit_layout == TRUE
    {
        comp_options.vulkan.supports_explicit_smem = true;
    }

    if let Some(maintenance_9) = &extended_feat.maintenance_9
        && maintenance_9.maintenance9 == TRUE
    {
        comp_options.vulkan.supports_arbitrary_bitwise = true;
    }

    if let Some(index_64) = &extended_feat.index_64
        && index_64.shader64_bit_indexing == TRUE
    {
        let instance = adapter.shared_instance().raw_instance();
        let device = adapter.raw_physical_device();
        let memory_props = unsafe { instance.get_physical_device_memory_properties(device) };
        let num_heaps = memory_props.memory_heap_count as usize;
        let mut heaps = memory_props.memory_heaps.iter().take(num_heaps);
        if let Some(heap) = heaps.find(|it| it.flags.contains(MemoryHeapFlags::DEVICE_LOCAL)) {
            let heap_size = heap.size;
            let max_page_size = match memory_config {
                #[cfg(not(exclusive_memory_only))]
                MemoryConfiguration::SubSlices => heap_size / 4,
                MemoryConfiguration::ExclusivePages => heap_size,
                MemoryConfiguration::Custom { .. } => heap_size,
            };
            props.memory.max_page_size = max_page_size;
        }
    }

    if extended_feat.cooperative_matrix.is_some() {
        register_cmma(ash, adapter, props);
    }

    true
}

fn register_types(props: &mut DeviceProperties, ext_feat: &ExtendedFeatures<'_>) {
    use cubecl_core::ir::{ElemType, FloatKind, IntKind, StorageType};

    props.register_address_type(AddressType::U32);

    if let Some(index_64) = &ext_feat.index_64
        && index_64.shader64_bit_indexing == TRUE
    {
        props.register_address_type(AddressType::U64);
    }

    let default_types = [
        ElemType::UInt(UIntKind::U32),
        ElemType::Int(IntKind::I32),
        ElemType::Float(FloatKind::F32),
        ElemType::Bool,
    ];

    let default_atomic_types = [ElemType::Int(IntKind::I32), ElemType::UInt(UIntKind::U32)];

    let storage16 = ext_feat
        .buf_16
        .is_some_and(|it| it.uniform_and_storage_buffer16_bit_access == TRUE);
    let storage8 = ext_feat
        .buf_16
        .is_some_and(|it| it.uniform_and_storage_buffer16_bit_access == TRUE);

    for ty in default_types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in default_atomic_types {
        props.register_atomic_type_usage(Type::new(StorageType::Atomic(ty)), AtomicUsage::all());
    }

    if ext_feat.core.shader_int64 == TRUE {
        props.register_type_usage(ElemType::UInt(UIntKind::U64), TypeUsage::all());
        props.register_type_usage(ElemType::Int(IntKind::I64), TypeUsage::all());
    }

    if ext_feat.core.shader_int16 == TRUE {
        props.register_type_usage(
            ElemType::UInt(UIntKind::U16),
            TypeUsage::maybe_store(storage16),
        );
        props.register_type_usage(
            ElemType::Int(IntKind::I16),
            TypeUsage::maybe_store(storage16),
        );
    }

    if ext_feat.core.shader_float64 == TRUE {
        props.register_type_usage(ElemType::Float(FloatKind::F64), TypeUsage::all());
    }

    if let Some(atomic_int64) = ext_feat.atomic_int64
        && atomic_int64.shader_buffer_int64_atomics == TRUE
    {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Int(IntKind::I64))),
            AtomicUsage::all(),
        );
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U64))),
            AtomicUsage::all(),
        );
    }

    if let Some(float16_int8) = ext_feat.float16_int8 {
        if float16_int8.shader_float16 == TRUE {
            props.register_type_usage(
                ElemType::Float(FloatKind::F16),
                TypeUsage::maybe_store(storage16),
            );
        }
        if float16_int8.shader_int8 == TRUE {
            props.register_type_usage(ElemType::Int(IntKind::I8), TypeUsage::maybe_store(storage8));
            props.register_type_usage(
                ElemType::UInt(UIntKind::U8),
                TypeUsage::maybe_store(storage8),
            );
        }
    }

    if let Some(bfloat16) = ext_feat.bfloat16 {
        if bfloat16.shader_b_float16_type == TRUE {
            props.register_type_usage(ElemType::Float(FloatKind::BF16), TypeUsage::Conversion);
            if storage16 {
                props.register_type_usage(ElemType::Float(FloatKind::BF16), TypeUsage::Buffer);
            }
        }
        if bfloat16.shader_b_float16_dot_product == TRUE {
            props.register_type_usage(ElemType::Float(FloatKind::BF16), TypeUsage::DotProduct);
        }
    }

    if let Some(float8) = ext_feat.float8
        && float8.shader_float8 == TRUE
    {
        props.register_type_usage(ElemType::Float(FloatKind::E4M3), TypeUsage::Conversion);
        props.register_type_usage(ElemType::Float(FloatKind::E5M2), TypeUsage::Conversion);
        if storage8 {
            props.register_type_usage(ElemType::Float(FloatKind::E4M3), TypeUsage::Buffer);
            props.register_type_usage(ElemType::Float(FloatKind::E5M2), TypeUsage::Buffer);
        }
    }

    if let Some(atomic_float) = ext_feat.atomic_float {
        if atomic_float.shader_buffer_float32_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float32_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
                AtomicUsage::Add,
            );
        }
        if atomic_float.shader_buffer_float64_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F64))),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float64_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F64))),
                AtomicUsage::Add,
            );
        }
    }

    if let Some(atomic_float) = ext_feat.atomic_float2 {
        if atomic_float.shader_buffer_float16_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float16_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))),
                AtomicUsage::Add,
            );
        }
        if atomic_float.shader_buffer_float16_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))),
                AtomicUsage::MinMax,
            );
        }
        if atomic_float.shader_buffer_float32_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))),
                AtomicUsage::MinMax,
            );
        }
        if atomic_float.shader_buffer_float64_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F64))),
                AtomicUsage::MinMax,
            );
        }
    }

    // Hardcoded support for f16 vectorized to 2 or 4 elements
    if let Some(nv_atomic_float_vector) = ext_feat.nv_atomic_float_vector
        && nv_atomic_float_vector.shader_float16_vector_atomics == TRUE
    {
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))).with_vector_size(2),
            AtomicUsage::all(),
        );
        props.register_atomic_type_usage(
            Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F16))).with_vector_size(4),
            AtomicUsage::all(),
        );
    }
}

fn register_cmma(ash: &InstanceShared, adapter: &vulkan::Adapter, props: &mut DeviceProperties) {
    let cmma = cooperative_matrix::Instance::new(ash.entry(), ash.raw_instance());
    let num_elems = unsafe {
        cmma.get_physical_device_cooperative_matrix_properties_len(
            adapter.raw_physical_device().into(),
        )
        .unwrap()
    };
    let mut properties = vec![Default::default(); num_elems];
    unsafe {
        cmma.get_physical_device_cooperative_matrix_properties(
            adapter.raw_physical_device().into(),
            &mut properties,
        )
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
            let mut min_current = props.hardware.min_tensor_cores_dim.unwrap_or(it.m_size);
            min_current = u32::min(min_current, it.m_size);
            min_current = u32::min(min_current, it.n_size);
            min_current = u32::min(min_current, it.k_size);
            props.hardware.min_tensor_cores_dim = Some(min_current);

            Some(MmaConfig {
                a_type: convert_type(it.a_type)?.into(),
                b_type: convert_type(it.b_type)?.into(),
                cd_type: convert_type(it.c_type)?.into(),
                m: it.m_size,
                k: it.k_size,
                n: it.n_size,
            })
        })
        .collect::<Vec<_>>();
    log::debug!("Supported CMMA sizes: {sizes:#?}");

    for size in sizes {
        props.features.matmul.cmma.insert(size);
    }
}

fn convert_type(vk_ty: ComponentTypeKHR) -> Option<ElemType> {
    let ty = match vk_ty {
        ComponentTypeKHR::FLOAT8_E4M3_EXT => ElemType::Float(FloatKind::E4M3),
        ComponentTypeKHR::FLOAT8_E5M2_EXT => ElemType::Float(FloatKind::E5M2),
        ComponentTypeKHR::FLOAT16 => ElemType::Float(FloatKind::F16),
        ComponentTypeKHR::BFLOAT16 => ElemType::Float(FloatKind::BF16),
        ComponentTypeKHR::FLOAT32 => ElemType::Float(FloatKind::F32),
        ComponentTypeKHR::FLOAT64 => ElemType::Float(FloatKind::F64),
        ComponentTypeKHR::SINT8 => ElemType::Int(IntKind::I8),
        ComponentTypeKHR::SINT16 => ElemType::Int(IntKind::I16),
        ComponentTypeKHR::SINT32 => ElemType::Int(IntKind::I32),
        ComponentTypeKHR::SINT64 => ElemType::Int(IntKind::I64),
        ComponentTypeKHR::UINT8 => ElemType::UInt(UIntKind::U8),
        ComponentTypeKHR::UINT16 => ElemType::UInt(UIntKind::U16),
        ComponentTypeKHR::UINT32 => ElemType::UInt(UIntKind::U32),
        ComponentTypeKHR::UINT64 => ElemType::UInt(UIntKind::U64),
        _ => None?,
    };
    Some(ty)
}

/// Check robustness, compile, and optionally dump SPIR-V
pub(crate) fn compile(
    dyn_comp: &mut AutoCompiler,
    server: &mut WgpuServer,
    kernel: <WgpuServer as ComputeServer>::Kernel,
    mode: ExecutionMode,
) -> Result<CompiledKernel<AutoCompiler>, CompilationError> {
    log::debug!("Compiling {}", kernel.name());
    let compiled = kernel.compile(
        dyn_comp,
        &server.compilation_options,
        mode,
        kernel.address_type(),
    )?;
    #[cfg(feature = "spirv-dump")]
    dump_spirv(&compiled, kernel.name(), kernel.id());
    Ok(compiled)
}

#[cfg(feature = "spirv-dump")]
fn dump_spirv(
    compiled: &CompiledKernel<AutoCompiler>,
    name: &str,
    id: cubecl_runtime::id::KernelId,
) {
    use std::{
        fs,
        hash::{DefaultHasher, Hash, Hasher},
    };

    if let Ok(dir) = std::env::var("CUBECL_DEBUG_SPIRV")
        && let Some(repr) = compiled.repr.as_ref().and_then(|repr| repr.as_spirv())
    {
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
        let kernel = &repr.assembled_module;
        let kernel = kernel
            .iter()
            .flat_map(|it| it.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(format!("{dir}/{name}.spv"), kernel).unwrap();
        if let Some(optimizer) = &repr.optimizer {
            fs::write(format!("{dir}/{name}.ir.txt"), format!("{}", optimizer)).unwrap();
            fs::write(
                format!("{dir}/{name}.ir.dot"),
                format!("{}", optimizer.dot_viz()),
            )
            .unwrap();
        }
    }
}
