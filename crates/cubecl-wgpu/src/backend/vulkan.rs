use ash::{
    khr::cooperative_matrix,
    vk::{ComponentTypeKHR, DeviceCreateInfo, DeviceQueueCreateInfo, ScopeKHR, TRUE},
};
use cubecl_core::{
    compute::Visibility,
    ir::{Elem, FloatKind, IntKind, UIntKind},
    AtomicFeature, Feature, WgpuCompilationOptions,
};
use cubecl_runtime::DeviceProperties;
use cubecl_spirv::SpirvKernel;
use features::ExtendedFeatures;
use wgpu::{
    hal::{
        self,
        vulkan::{self, InstanceShared},
    },
    DeviceDescriptor, Features, Limits,
};

mod features;

pub fn bindings(repr: &SpirvKernel) -> Vec<(usize, Visibility)> {
    repr.bindings
        .iter()
        .enumerate()
        .map(|it| (it.0, it.1.visibility))
        .collect()
}

pub async fn request_vulkan_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let limits = adapter.limits();
    let features = adapter.features();
    unsafe {
        adapter.as_hal::<hal::api::Vulkan, _, _>(|hal_adapter| {
            request_device(adapter, hal_adapter.unwrap(), features, limits)
        })
    }
}

pub fn register_vulkan_features(
    adapter: &wgpu::Adapter,
    props: &mut cubecl_runtime::DeviceProperties<cubecl_core::Feature>,
    comp_options: &mut WgpuCompilationOptions,
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

/// Request device with required features, plus CMMA if available.
fn request_device(
    wgpu_adapter: &wgpu::Adapter,
    adapter: &vulkan::Adapter,
    mut features: Features,
    limits: Limits,
) -> (wgpu::Device, wgpu::Queue) {
    let full_feat = features;
    // This registers only f16 but not u8/i8, so remove so we can manually add them
    features.remove(Features::SHADER_F16);
    // Skip float features since we already register a more general version manually
    features.remove(Features::SHADER_FLOAT32_ATOMIC);

    let ash = adapter.shared_instance();
    let mut extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);
    let extensions = adapter.required_device_extensions(features);
    let mut phys_features = adapter.physical_device_features(&extensions, features);

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
                full_feat,
                &wgpu::MemoryHints::MemoryUsage,
                family_info.queue_family_index,
                0,
            )
            .expect("Failed to create HAL device")
    };

    let descriptor = DeviceDescriptor {
        label: None,
        required_features: full_feat,
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
    comp_options: &mut WgpuCompilationOptions,
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
