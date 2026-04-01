use std::{ffi::CStr, ptr::null_mut};

use tracel_ash::vk::*;
use wgpu::{Features, hal::vulkan};

#[derive(Default, Debug)]
pub struct ExtendedFeatures<'a> {
    pub core: PhysicalDeviceFeatures,

    // Required
    pub vulkan_memory_model: Option<PhysicalDeviceVulkanMemoryModelFeatures<'a>>,
    pub buffer_device_address: Option<PhysicalDeviceBufferDeviceAddressFeatures<'a>>,

    // Optional core
    pub atomic_int64: Option<PhysicalDeviceShaderAtomicInt64Features<'a>>,
    pub float16_int8: Option<PhysicalDeviceShaderFloat16Int8Features<'a>>,
    pub buf_16: Option<PhysicalDevice16BitStorageFeatures<'a>>,
    pub buf_8: Option<PhysicalDevice8BitStorageFeatures<'a>>,
    pub subgroup_extended: Option<PhysicalDeviceShaderSubgroupExtendedTypesFeatures<'a>>,
    pub uniform_standard_layout: Option<PhysicalDeviceUniformBufferStandardLayoutFeatures<'a>>,

    // extensions
    pub cooperative_matrix: Option<PhysicalDeviceCooperativeMatrixFeaturesKHR<'a>>,
    pub atomic_float: Option<PhysicalDeviceShaderAtomicFloatFeaturesEXT<'a>>,
    pub atomic_float2: Option<PhysicalDeviceShaderAtomicFloat2FeaturesEXT<'a>>,
    pub float_controls2: Option<PhysicalDeviceShaderFloatControls2FeaturesKHR<'a>>,
    pub bfloat16: Option<PhysicalDeviceShaderBfloat16FeaturesKHR<'a>>,
    pub float8: Option<PhysicalDeviceShaderFloat8FeaturesEXT<'a>>,

    pub wg_explicit_layout: Option<PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR<'a>>,
    pub index_64: Option<PhysicalDeviceShader64BitIndexingFeaturesEXT<'a>>,
    pub uniform_unsized_array: Option<PhysicalDeviceShaderUniformBufferUnsizedArrayFeaturesEXT<'a>>,
    /// Only used to enable the spec compliant rem/mod behavior
    pub maintenance_8: Option<PhysicalDeviceMaintenance8FeaturesKHR<'a>>,
    pub maintenance_9: Option<PhysicalDeviceMaintenance9FeaturesKHR<'a>>,

    // Nvidia
    pub nv_atomic_float_vector: Option<PhysicalDeviceShaderAtomicFloat16VectorFeaturesNV<'a>>,

    pub max_spirv_version: (u8, u8),
    pub extensions: Vec<&'static CStr>,
}

macro_rules! fill_core {
    ($self: expr, $caps: expr, $($extension: expr; $version: expr => $field: ident,)*) => {
        $(if $caps.properties().api_version >= $version {
            $self.$field = Some(Default::default());
        } else if $caps.supports_extension($extension) {
            $self.extensions.push($extension);
            $self.$field = Some(Default::default());
        })*
    };
}

macro_rules! fill_opt {
    ($self: expr, $caps: expr, $($extension: expr => $field: ident,)*) => {
        $(if $caps.supports_extension($extension) {
            $self.extensions.push($extension);
            $self.$field = Some(Default::default());
        })*
    };
}

macro_rules! zero_opt {
    ($self: expr, $($name: ident,)*) => {
        $(if let Some($name) = &mut $self.$name {
            $name.p_next = null_mut();
        })*
    };
}

impl<'a> ExtendedFeatures<'a> {
    pub fn from_adapter(
        ash: &ash::Instance,
        adapter: &vulkan::Adapter,
        features: Features,
    ) -> Self {
        let mut this = Self::default();
        this.fill_extensions(adapter, features);
        this.fill_features(ash, adapter);
        this
    }

    fn fill_extensions(&mut self, adapter: &vulkan::Adapter, features: Features) {
        self.extensions = adapter.required_device_extensions(features);

        let phys_caps = adapter.physical_device_capabilities();
        let version = phys_caps.properties().api_version;

        if version >= API_VERSION_1_3 {
            self.max_spirv_version = (1, 6);
        } else if version >= API_VERSION_1_2 {
            self.max_spirv_version = (1, 5);
        } else if phys_caps.supports_extension(KHR_SPIRV_1_4_NAME) {
            self.extensions.push(KHR_SHADER_FLOAT_CONTROLS_NAME);
            self.extensions.push(KHR_SPIRV_1_4_NAME);
            self.max_spirv_version = (1, 4);
        } else if version >= API_VERSION_1_1 {
            self.max_spirv_version = (1, 3);
        } else {
            self.extensions.push(KHR_STORAGE_BUFFER_STORAGE_CLASS_NAME);
            self.max_spirv_version = (1, 0);
        }

        fill_core!(
            self,
            phys_caps,
            KHR_VULKAN_MEMORY_MODEL_NAME; API_VERSION_1_2 => vulkan_memory_model,
            KHR_BUFFER_DEVICE_ADDRESS_NAME; API_VERSION_1_2 => buffer_device_address,
            KHR_SHADER_ATOMIC_INT64_NAME; API_VERSION_1_2 => atomic_int64,
            KHR_SHADER_FLOAT16_INT8_NAME; API_VERSION_1_2 => float16_int8,
            KHR_16BIT_STORAGE_NAME; API_VERSION_1_1 => buf_16,
            KHR_8BIT_STORAGE_NAME; API_VERSION_1_2 => buf_8,
            KHR_SHADER_SUBGROUP_EXTENDED_TYPES_NAME; API_VERSION_1_2 => subgroup_extended,
            KHR_UNIFORM_BUFFER_STANDARD_LAYOUT_NAME; API_VERSION_1_2 => uniform_standard_layout,
        );

        fill_opt!(self,
            phys_caps,
            KHR_COOPERATIVE_MATRIX_NAME => cooperative_matrix,
            EXT_SHADER_ATOMIC_FLOAT_NAME => atomic_float,
            EXT_SHADER_ATOMIC_FLOAT2_NAME => atomic_float2,
            KHR_SHADER_FLOAT_CONTROLS2_NAME => float_controls2,
            KHR_SHADER_BFLOAT16_NAME => bfloat16,
            EXT_SHADER_FLOAT8_NAME => float8,
            KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_NAME => wg_explicit_layout,
            EXT_SHADER_64BIT_INDEXING_NAME => index_64,
            EXT_SHADER_UNIFORM_BUFFER_UNSIZED_ARRAY_NAME => uniform_unsized_array,
            KHR_MAINTENANCE8_NAME => maintenance_8,
            KHR_MAINTENANCE9_NAME => maintenance_9,
            NV_SHADER_ATOMIC_FLOAT16_VECTOR_NAME => nv_atomic_float_vector,
        );
    }

    pub fn add_to_device_create(
        &'a mut self,
        mut info: DeviceCreateInfo<'a>,
    ) -> DeviceCreateInfo<'a> {
        fn push_opt<'a, T: Extends<DeviceCreateInfo<'a>> + TaggedStructure<'a>>(
            mut info: DeviceCreateInfo<'a>,
            feat: &'a mut Option<T>,
        ) -> DeviceCreateInfo<'a> {
            if let Some(feat) = feat {
                info = info.push_or_update(feat);
            }
            info
        }

        // Required
        info = push_opt(info, &mut self.vulkan_memory_model);
        info = push_opt(info, &mut self.buffer_device_address);

        // Core
        info = push_opt(info, &mut self.atomic_int64);
        info = push_opt(info, &mut self.float16_int8);
        info = push_opt(info, &mut self.buf_16);
        info = push_opt(info, &mut self.buf_8);
        info = push_opt(info, &mut self.subgroup_extended);
        info = push_opt(info, &mut self.uniform_standard_layout);

        // Extensions
        info = push_opt(info, &mut self.cooperative_matrix);
        info = push_opt(info, &mut self.atomic_float);
        info = push_opt(info, &mut self.atomic_float2);
        info = push_opt(info, &mut self.float_controls2);
        info = push_opt(info, &mut self.bfloat16);
        info = push_opt(info, &mut self.float8);
        info = push_opt(info, &mut self.wg_explicit_layout);
        info = push_opt(info, &mut self.index_64);
        info = push_opt(info, &mut self.uniform_unsized_array);
        info = push_opt(info, &mut self.maintenance_8);
        info = push_opt(info, &mut self.maintenance_9);

        // Nvidia
        info = push_opt(info, &mut self.nv_atomic_float_vector);

        info
    }

    fn fill_features(&mut self, ash: &ash::Instance, adapter: &vulkan::Adapter) {
        let mut features = PhysicalDeviceFeatures2::default();

        fn push_opt<'a, 'b: 'a, T: Extends<PhysicalDeviceFeatures2<'a>> + TaggedStructure<'b>>(
            mut features: PhysicalDeviceFeatures2<'a>,
            feat: &'a mut Option<T>,
        ) -> PhysicalDeviceFeatures2<'a> {
            if let Some(feat) = feat {
                features = features.push(feat);
            }
            features
        }

        // Required
        features = push_opt(features, &mut self.vulkan_memory_model);
        features = push_opt(features, &mut self.buffer_device_address);

        // Core
        features = push_opt(features, &mut self.atomic_int64);
        features = push_opt(features, &mut self.float16_int8);
        features = push_opt(features, &mut self.buf_16);
        features = push_opt(features, &mut self.buf_8);
        features = push_opt(features, &mut self.subgroup_extended);
        features = push_opt(features, &mut self.uniform_standard_layout);

        // Extensions
        features = push_opt(features, &mut self.cooperative_matrix);
        features = push_opt(features, &mut self.atomic_float);
        features = push_opt(features, &mut self.atomic_float2);
        features = push_opt(features, &mut self.float_controls2);
        features = push_opt(features, &mut self.bfloat16);
        features = push_opt(features, &mut self.float8);
        features = push_opt(features, &mut self.wg_explicit_layout);
        features = push_opt(features, &mut self.index_64);
        features = push_opt(features, &mut self.uniform_unsized_array);
        features = push_opt(features, &mut self.maintenance_8);
        features = push_opt(features, &mut self.maintenance_9);

        // Nvidia
        features = push_opt(features, &mut self.nv_atomic_float_vector);

        unsafe {
            // convert to ash version, they represent the same type so this is safe
            let features =
                &mut *<*mut _>::cast::<ash::vk::PhysicalDeviceFeatures2<'_>>(&mut features);
            ash.get_physical_device_features2(adapter.raw_physical_device(), features);
            self.core = core::mem::transmute::<
                ash::vk::PhysicalDeviceFeatures,
                tracel_ash::vk::PhysicalDeviceFeatures,
            >(features.features);
        }

        self.zero_pointers();
    }

    /// Leaving these set seems to cause misaligned deref
    fn zero_pointers(&mut self) {
        zero_opt!(
            self,
            // Required
            vulkan_memory_model,
            buffer_device_address,
            // Core
            atomic_int64,
            float16_int8,
            buf_16,
            buf_8,
            subgroup_extended,
            uniform_standard_layout,
            // Extensions
            cooperative_matrix,
            atomic_float,
            atomic_float2,
            float_controls2,
            bfloat16,
            float8,
            wg_explicit_layout,
            index_64,
            uniform_unsized_array,
            maintenance_8,
            maintenance_9,
            nv_atomic_float_vector,
        );
    }

    pub fn has_required_features(&self) -> bool {
        let mem_model = self.vulkan_memory_model.is_some_and(|it| {
            it.vulkan_memory_model == TRUE && it.vulkan_memory_model_device_scope == TRUE
        });
        let device_address = self
            .buffer_device_address
            .is_some_and(|it| it.buffer_device_address == TRUE);
        let storage_class = self.max_spirv_version >= (1, 3)
            || self
                .extensions
                .contains(&KHR_STORAGE_BUFFER_STORAGE_CLASS_NAME);
        mem_model && device_address && storage_class
    }
}

trait InfoExt<'a>: Sized + TaggedStructure<'a> + 'a {
    fn push_or_update<T: Extends<Self> + TaggedStructure<'a>>(self, feat: &'a mut T) -> Self;
}

impl<'a> InfoExt<'a> for DeviceCreateInfo<'a> {
    fn push_or_update<T: Extends<Self> + TaggedStructure<'a>>(mut self, feat: &'a mut T) -> Self {
        let this = &mut self as *mut DeviceCreateInfo<'a>;
        let mut this = unsafe { &mut *this.cast::<BaseOutStructure<'a>>() };
        while !this.p_next.is_null() {
            let structure = unsafe { &mut *this.p_next };
            if structure.s_type == T::STRUCTURE_TYPE {
                let feat_ptr = (feat as *mut T).cast::<BaseOutStructure<'a>>();
                let feat = unsafe { &mut *feat_ptr };

                this.p_next = feat_ptr;
                feat.p_next = structure.p_next;
                return self;
            }
            this = structure;
        }

        self.push(feat)
    }
}
