use cubecl_core::{
    ExecutionMode, WgpuCompilationOptions,
    ir::{AddressType, ElemType, FloatKind, IntKind, UIntKind},
    prelude::{CompiledKernel, Visibility},
    server::ComputeServer,
};
use cubecl_ir::{DeviceProperties, features::*};
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

pub fn bindings(repr: &SpirvKernel) -> (Vec<Visibility>, Vec<Visibility>) {
    let bindings: Vec<_> = repr.bindings.iter().map(|it| it.visibility).collect();
    let mut meta = vec![];
    if repr.has_metadata {
        meta.push(Visibility::Read);
    }
    meta.extend(repr.scalars.iter().map(|_| Visibility::Read));
    (bindings, meta)
}

pub async fn request_vulkan_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
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
) {
    let features = adapter.features();
    unsafe {
        if let Some(adapter) = adapter.as_hal::<hal::api::Vulkan>() {
            register_features(&adapter, props, features, comp_options);
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
                full_feat,
                &memory_hints,
                family_info.queue_family_index,
                0,
            )
            .expect("Failed to create HAL device")
    };

    let descriptor = DeviceDescriptor {
        label: None,
        required_features: full_feat,
        required_limits: limits,
        memory_hints,
        trace: wgpu::Trace::Off,
    };

    unsafe {
        wgpu_adapter
            .create_device_from_hal(device, &descriptor)
            .expect("Failed to create wgpu device")
    }
}

/// Request device's supported features
fn register_features(
    adapter: &vulkan::Adapter,
    props: &mut DeviceProperties,
    features: Features,
    comp_options: &mut WgpuCompilationOptions,
) {
    let ash = adapter.shared_instance();
    let extended_feat = ExtendedFeatures::from_adapter(ash.raw_instance(), adapter, features);

    log::debug!("Supported Vulkan features: {extended_feat:#?}");

    register_types(props, &extended_feat);
    comp_options.supports_u64 = true;
    props.features.plane.insert(Plane::Sync);

    if let Some(float_controls2) = &extended_feat.float_controls2
        && float_controls2.shader_float_controls2 == TRUE
    {
        comp_options.supports_fp_fast_math = true;
    }

    if let Some(wg_explicit_layout) = &extended_feat.wg_explicit_layout
        && wg_explicit_layout.workgroup_memory_explicit_layout == TRUE
    {
        comp_options.supports_explicit_smem = true;
    }

    if extended_feat.cmma.is_some() {
        register_cmma(ash, adapter, props);
    }
}

fn register_types(props: &mut DeviceProperties, ext_feat: &ExtendedFeatures<'_>) {
    use cubecl_core::ir::{ElemType, FloatKind, IntKind, StorageType};

    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let mut register = |elem: StorageType, usage: EnumSet<TypeUsage>| {
        props.register_type_usage(elem, usage);
    };

    let default_types = [
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::Float(FloatKind::F32),
        // Elem::Float(FloatKind::F64),
        ElemType::Bool,
    ];

    let default_atomic_types = [
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
    ];

    for ty in default_types {
        register(ty.into(), TypeUsage::all_scalar());
    }

    for ty in default_atomic_types {
        register(StorageType::Atomic(ty), TypeUsage::all_atomic())
    }

    if ext_feat.float16_int8.shader_float16 == TRUE {
        register(
            ElemType::Float(FloatKind::F16).into(),
            TypeUsage::all_scalar(),
        );
    }
    if ext_feat.float16_int8.shader_int8 == TRUE {
        register(ElemType::Int(IntKind::I8).into(), TypeUsage::all_scalar());
        register(ElemType::UInt(UIntKind::U8).into(), TypeUsage::all_scalar());
    }

    if let Some(bfloat16) = ext_feat.bfloat16 {
        if bfloat16.shader_b_float16_type == TRUE {
            register(
                ElemType::Float(FloatKind::BF16).into(),
                TypeUsage::Conversion | TypeUsage::Buffer,
            );
        }
        if bfloat16.shader_b_float16_dot_product == TRUE {
            register(
                ElemType::Float(FloatKind::BF16).into(),
                TypeUsage::DotProduct.into(),
            );
        }
    }

    if let Some(float8) = ext_feat.float8
        && float8.shader_float8 == TRUE
    {
        register(
            ElemType::Float(FloatKind::E4M3).into(),
            TypeUsage::Conversion | TypeUsage::Buffer,
        );
        register(
            ElemType::Float(FloatKind::E5M2).into(),
            TypeUsage::Conversion | TypeUsage::Buffer,
        );
    }

    if let Some(atomic_float) = ext_feat.atomic_float {
        if atomic_float.shader_buffer_float32_atomics == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F32)),
                TypeUsage::AtomicLoadStore.into(),
            );
        }
        if atomic_float.shader_buffer_float32_atomic_add == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F32)),
                TypeUsage::AtomicAdd.into(),
            );
        }
        if atomic_float.shader_buffer_float64_atomics == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F64)),
                TypeUsage::AtomicLoadStore.into(),
            );
        }
        if atomic_float.shader_buffer_float64_atomic_add == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F64)),
                TypeUsage::AtomicAdd.into(),
            );
        }
    }

    if let Some(atomic_float) = ext_feat.atomic_float2 {
        if atomic_float.shader_buffer_float16_atomics == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F16)),
                TypeUsage::AtomicLoadStore.into(),
            );
        }
        if atomic_float.shader_buffer_float16_atomic_add == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F16)),
                TypeUsage::AtomicAdd.into(),
            );
        }
        if atomic_float.shader_buffer_float16_atomic_min_max == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F16)),
                TypeUsage::AtomicMinMax.into(),
            );
        }
        if atomic_float.shader_buffer_float32_atomic_min_max == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F32)),
                TypeUsage::AtomicMinMax.into(),
            );
        }
        if atomic_float.shader_buffer_float64_atomic_min_max == TRUE {
            register(
                StorageType::Atomic(ElemType::Float(FloatKind::F64)),
                TypeUsage::AtomicMinMax.into(),
            );
        }
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
        props.features.cmma.insert(size);
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
        let kernel = repr.assemble().into_iter();
        let kernel = kernel.flat_map(|it| it.to_le_bytes()).collect::<Vec<_>>();
        fs::write(format!("{dir}/{name}.spv"), kernel).unwrap();
        fs::write(
            format!("{dir}/{name}.ir.txt"),
            format!("{}", repr.optimizer),
        )
        .unwrap();
        fs::write(
            format!("{dir}/{name}.ir.dot"),
            format!("{}", repr.optimizer.dot_viz()),
        )
        .unwrap();
    }
}
