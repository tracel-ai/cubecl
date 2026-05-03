use alloc::collections::BTreeSet;

use ash::vk::{
    self, API_VERSION_1_1, BufferDeviceAddressInfo, BufferUsageFlags,
    KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME, MemoryAllocateFlags, MemoryAllocateFlagsInfo,
    MemoryAllocateInfo, MemoryHeap, MemoryHeapFlags, MemoryPropertyFlags, PhysicalDevice,
    SharingMode,
};
use cubecl_core::{
    ExecutionMode, MemoryConfiguration, WgpuCompilationOptions,
    backtrace::BackTrace,
    ir::{AddressType, ElemType, FloatKind, IntKind, UIntKind},
    prelude::{CompiledKernel, Visibility},
    server::{ComputeServer, IoError, KernelArguments},
};
use cubecl_ir::{DeviceProperties, Type, features::*};
use cubecl_runtime::compiler::CompilationError;
use cubecl_spirv::{GLCompute, SpirvCompiler, SpirvKernel};
use features::ExtendedFeatures;
use tracel_ash::{
    khr::cooperative_matrix,
    nv::cooperative_matrix2,
    vk::{
        ComponentTypeKHR, DeviceCreateInfo, DeviceQueueCreateInfo,
        PhysicalDeviceCooperativeMatrix2PropertiesNV, PhysicalDeviceProperties2, ScopeKHR, TRUE,
        TaggedStructure,
    },
};
use wgpu::{
    BufferUsages, BufferUses, DeviceDescriptor, Features, Limits,
    hal::{
        self,
        vulkan::{self, InstanceShared},
    },
};

use crate::{AutoCompiler, WgpuServer};

mod features;

pub type VkSpirvCompiler = SpirvCompiler<GLCompute>;

pub fn bindings(repr: &SpirvKernel, _bindings: &KernelArguments) -> (Vec<Visibility>, usize) {
    match repr.immediate_size {
        Some(immediate_size) => (vec![], immediate_size),
        None => (vec![Visibility::Uniform], 0),
    }
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
    let properties = adapter.physical_device_capabilities().properties();
    let mut phys_features = adapter.physical_device_features(&extensions, features);

    // This is clamped to 256 because of Naga validation reasons, but we only use it for Vulkan.
    // So ignore the clamping and use the full value.
    // This is mainly relevant on M-series Macs where the max size is 4096 (much larger than the 256 wgpu clamps to).
    limits.max_immediate_size = properties.limits.max_push_constants_size;

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

pub(crate) fn create_storage_buffer(
    wgpu_device: &wgpu::Device,
    desc: &wgpu::BufferDescriptor,
) -> Result<(wgpu::Buffer, u64), IoError> {
    let device: &vulkan::Device = unsafe { &wgpu_device.as_hal::<hal::api::Vulkan>().unwrap() };
    let instance = device.shared_instance().raw_instance();
    let phys_device = device.raw_physical_device();
    let device = device.raw_device();

    let uses = map_buffer_usage(desc.usage);
    let usage_flags = wgpu::hal::vulkan::conv::map_buffer_usage(uses);

    let vk_info = ash::vk::BufferCreateInfo::default()
        .size(desc.size)
        .usage(usage_flags | BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(SharingMode::EXCLUSIVE);

    let buffer = unsafe {
        device
            .create_buffer(&vk_info, None)
            .map_err(|err| as_io_error(err, desc.size))?
    };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_props = unsafe { instance.get_physical_device_memory_properties(phys_device) };
    let num_types = memory_props.memory_type_count as usize;
    let memory_types = memory_props.memory_types.iter().take(num_types);

    let (memory_type_idx, _) = memory_types
        .enumerate()
        .filter(|(i, _)| requirements.memory_type_bits & (1 << *i) != 0)
        .find(|(_, it)| {
            it.property_flags
                .contains(MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .ok_or_else(|| IoError::Unknown {
            description: "No device local heap found".into(),
            backtrace: BackTrace::capture(),
        })?;

    let mut alloc_flags =
        MemoryAllocateFlagsInfo::default().flags(MemoryAllocateFlags::DEVICE_ADDRESS);

    let alloc_info = MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_idx as u32)
        .push_next(&mut alloc_flags);

    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .map_err(|err| as_io_error(err, desc.size))?
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .map_err(|err| as_io_error(err, desc.size))?
    };

    let addr_info = BufferDeviceAddressInfo::default().buffer(buffer);
    let device_address = unsafe { device.get_buffer_device_address(&addr_info) };

    let buffer = unsafe {
        wgpu::hal::vulkan::Buffer::from_raw_managed(buffer, memory, 0, requirements.size)
    };
    let buffer = unsafe { wgpu_device.create_buffer_from_hal::<hal::api::Vulkan>(buffer, desc) };

    Ok((buffer, device_address))
}

fn as_io_error(result: vk::Result, size: u64) -> IoError {
    match result {
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            }
        }
        err => IoError::Unknown {
            description: err.to_string(),
            backtrace: BackTrace::capture(),
        },
    }
}

fn map_buffer_usage(usage: BufferUsages) -> BufferUses {
    let mut u = BufferUses::empty();
    u.set(BufferUses::MAP_READ, usage.contains(BufferUsages::MAP_READ));
    u.set(
        BufferUses::MAP_WRITE,
        usage.contains(BufferUsages::MAP_WRITE),
    );
    u.set(BufferUses::COPY_SRC, usage.contains(BufferUsages::COPY_SRC));
    u.set(BufferUses::COPY_DST, usage.contains(BufferUsages::COPY_DST));
    u.set(BufferUses::INDEX, usage.contains(BufferUsages::INDEX));
    u.set(BufferUses::VERTEX, usage.contains(BufferUsages::VERTEX));
    u.set(BufferUses::UNIFORM, usage.contains(BufferUsages::UNIFORM));
    u.set(
        BufferUses::STORAGE_READ_ONLY | BufferUses::STORAGE_READ_WRITE,
        usage.contains(BufferUsages::STORAGE),
    );
    u.set(BufferUses::INDIRECT, usage.contains(BufferUsages::INDIRECT));
    u.set(
        BufferUses::QUERY_RESOLVE,
        usage.contains(BufferUsages::QUERY_RESOLVE),
    );
    u.set(
        BufferUses::BOTTOM_LEVEL_ACCELERATION_STRUCTURE_INPUT,
        usage.contains(BufferUsages::BLAS_INPUT),
    );
    u.set(
        BufferUses::TOP_LEVEL_ACCELERATION_STRUCTURE_INPUT,
        usage.contains(BufferUsages::TLAS_INPUT),
    );
    u
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

    let properties = adapter.physical_device_capabilities().properties();
    let extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);

    if !extended_feat.has_required_features() {
        return false;
    }

    log::debug!("Supported Vulkan features: {extended_feat:#?}");

    register_types(props, &extended_feat);

    comp_options.supports_vulkan_compiler = true;
    comp_options.supports_u64 = extended_feat.core.shader_int64 == TRUE;
    comp_options.vulkan.max_spirv_version = extended_feat.max_spirv_version;
    comp_options.vulkan.max_vector_size = 4;
    comp_options.vulkan.push_constant_size = properties.limits.max_push_constants_size as usize;

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

    if let Some(long_vector) = &extended_feat.long_vector
        && let Some(long_vector_properties) = &extended_feat.long_vector_properties
        && long_vector.long_vector == TRUE
    {
        comp_options.vulkan.supports_long_vectors = true;
        comp_options.vulkan.max_vector_size = long_vector_properties.max_vector_components as usize;
        props.hardware.max_vector_size = long_vector_properties.max_vector_components as usize;
    }

    if let Some(maintenance_9) = &extended_feat.maintenance_9
        && maintenance_9.maintenance9 == TRUE
    {
        comp_options.vulkan.supports_arbitrary_bitwise = true;
    }

    if let Some(index_64) = &extended_feat.index_64
        && index_64.shader64_bit_indexing == TRUE
        && let Some(heap) =
            device_local_heap(adapter.shared_instance(), adapter.raw_physical_device())
    {
        let heap_size = heap.size;
        let max_page_size = match memory_config {
            #[cfg(not(exclusive_memory_only))]
            MemoryConfiguration::SubSlices => heap_size / 4,
            MemoryConfiguration::ExclusivePages => heap_size,
            MemoryConfiguration::Custom { .. } => heap_size,
        };
        props.memory.max_page_size = max_page_size;
    }

    if extended_feat.cooperative_matrix.is_some() {
        register_cmma(ash, adapter, props);
    }

    if let Some(nv_cooperative_matrix2) = extended_feat.nv_cooperative_matrix2 {
        if nv_cooperative_matrix2.cooperative_matrix_flexible_dimensions == TRUE
            && nv_cooperative_matrix2.cooperative_matrix_workgroup_scope == TRUE
        {
            register_cooperative_matrix2(ash, adapter, props);
        }
        props.features.matmul.cmma_tensor_addressing =
            nv_cooperative_matrix2.cooperative_matrix_tensor_addressing == TRUE;
    }

    true
}

fn device_local_heap(instance: &InstanceShared, device: PhysicalDevice) -> Option<MemoryHeap> {
    let memory_props = unsafe {
        instance
            .raw_instance()
            .get_physical_device_memory_properties(device)
    };
    let num_heaps = memory_props.memory_heap_count as usize;
    let mut heaps = memory_props.memory_heaps.iter().take(num_heaps);
    heaps
        .find(|it| it.flags.contains(MemoryHeapFlags::DEVICE_LOCAL))
        .copied()
}

fn register_types(props: &mut DeviceProperties, ext_feat: &ExtendedFeatures<'_>) {
    use cubecl_core::ir::{ElemType, FloatKind, IntKind};

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
        props.register_atomic_type_usage(Type::atomic(ty), AtomicUsage::all());
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
            Type::atomic(ElemType::Int(IntKind::I64)),
            AtomicUsage::all(),
        );
        props.register_atomic_type_usage(
            Type::atomic(ElemType::UInt(UIntKind::U64)),
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
                Type::atomic(ElemType::Float(FloatKind::F32)),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float32_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F32)),
                AtomicUsage::Add,
            );
        }
        if atomic_float.shader_buffer_float64_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F64)),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float64_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F64)),
                AtomicUsage::Add,
            );
        }
    }

    if let Some(atomic_float) = ext_feat.atomic_float2 {
        if atomic_float.shader_buffer_float16_atomics == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F16)),
                AtomicUsage::LoadStore,
            );
        }
        if atomic_float.shader_buffer_float16_atomic_add == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F16)),
                AtomicUsage::Add,
            );
        }
        if atomic_float.shader_buffer_float16_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F16)),
                AtomicUsage::MinMax,
            );
        }
        if atomic_float.shader_buffer_float32_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F32)),
                AtomicUsage::MinMax,
            );
        }
        if atomic_float.shader_buffer_float64_atomic_min_max == TRUE {
            props.register_atomic_type_usage(
                Type::atomic(ElemType::Float(FloatKind::F64)),
                AtomicUsage::MinMax,
            );
        }
    }

    // Hardcoded support for f16 vectorized to 2 or 4 elements
    if let Some(nv_atomic_float_vector) = ext_feat.nv_atomic_float_vector
        && nv_atomic_float_vector.shader_float16_vector_atomics == TRUE
    {
        props.register_atomic_type_usage(
            Type::atomic(Type::scalar(ElemType::Float(FloatKind::F16)).with_vector_size(2)),
            AtomicUsage::all(),
        );
        props.register_atomic_type_usage(
            Type::atomic(Type::scalar(ElemType::Float(FloatKind::F16)).with_vector_size(4)),
            AtomicUsage::all(),
        );
    }
}

fn register_cmma(ash: &InstanceShared, adapter: &vulkan::Adapter, props: &mut DeviceProperties) {
    let instance = cooperative_matrix::Instance::new(ash.entry(), ash.raw_instance());
    let num_elems = unsafe {
        instance
            .get_physical_device_cooperative_matrix_properties_len(
                adapter.raw_physical_device().into(),
            )
            .unwrap()
    };
    let mut properties = vec![Default::default(); num_elems];
    unsafe {
        instance
            .get_physical_device_cooperative_matrix_properties(
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

fn register_cooperative_matrix2(
    ash: &InstanceShared,
    adapter: &vulkan::Adapter,
    props: &mut DeviceProperties,
) {
    let instance = cooperative_matrix2::Instance::new(ash.entry(), ash.raw_instance());
    let num_elems = unsafe {
        instance.get_physical_device_cooperative_matrix_flexible_dimensions_properties_len(
            adapter.raw_physical_device().into(),
        )
    }
    .unwrap();
    let mut configs = vec![Default::default(); num_elems];
    unsafe {
        instance.get_physical_device_cooperative_matrix_flexible_dimensions_properties(
            adapter.raw_physical_device().into(),
            &mut configs,
        )
    }
    .unwrap();

    let mut properties = PhysicalDeviceCooperativeMatrix2PropertiesNV::default();
    let mut tmp_props = PhysicalDeviceProperties2::default().push(&mut properties);

    unsafe {
        // convert to ash version, they represent the same type so this is safe
        let props = &mut *<*mut _>::cast::<ash::vk::PhysicalDeviceProperties2<'_>>(&mut tmp_props);
        ash.raw_instance()
            .get_physical_device_properties2(adapter.raw_physical_device(), props);
    }

    let max_dim = properties.cooperative_matrix_flexible_dimensions_max_dimension;

    let configs = configs
        .into_iter()
        .filter(|cfg| cfg.scope == ScopeKHR::WORKGROUP)
        .filter(|cfg| cfg.c_type == cfg.result_type)
        .filter_map(|cfg| {
            Some(CubeMmaConfig {
                a_type: convert_type(cfg.a_type)?.into(),
                b_type: convert_type(cfg.b_type)?.into(),
                cd_type: convert_type(cfg.c_type)?.into(),
                m_granularity: cfg.m_granularity,
                m_max: max_dim,
                n_granularity: cfg.n_granularity,
                n_max: max_dim,
                k_granularity: cfg.k_granularity,
                k_max: max_dim,
                units_per_block: Some(cfg.workgroup_invocations),
            })
        })
        .collect::<BTreeSet<_>>();

    log::debug!("Supported cube MMA configurations: {configs:#?}");

    props.features.matmul.cube_mma = configs;
    props.hardware.cube_mma_reserved_shared_memory =
        properties.cooperative_matrix_workgroup_scope_reserved_shared_memory as usize;
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
        std::fs::create_dir_all(&dir).unwrap();
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
            fs::write(format!("{dir}/{name}.ir.dot"), optimizer.main.dot_viz()).unwrap();
        }
    }
}
