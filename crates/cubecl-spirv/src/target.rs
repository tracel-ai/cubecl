use cubecl_core::prelude::KernelArg;
use cubecl_opt::BufferVisibility;
use rspirv::{
    dr::Operand,
    spirv::{
        self, AddressingModel, Capability, Decoration, ExecutionMode, ExecutionModel, MemoryModel,
        StorageClass, Word,
    },
};
use std::{fmt::Debug, iter};

use crate::{SpirvCompiler, extensions::TargetExtensions, item::Item, lookups::Buffer};

pub trait SpirvTarget:
    TargetExtensions<Self> + Debug + Clone + Default + Send + Sync + 'static
{
    fn set_modes(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        main: Word,
        builtins: Vec<Word>,
        cube_dims: Vec<u32>,
    );
    fn generate_params(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        bindings: &[KernelArg],
        visibility: &[BufferVisibility],
    ) -> Vec<Buffer>;
    fn load_params(b: &mut SpirvCompiler<Self>);
    fn info_storage_class(b: &mut SpirvCompiler<Self>) -> StorageClass;
    fn params_storage_class(b: &mut SpirvCompiler<Self>, num_buffers: usize) -> StorageClass;

    fn set_kernel_name(&mut self, name: impl Into<String>);
}

#[derive(Clone)]
pub struct GLCompute {
    kernel_name: String,
}

impl Default for GLCompute {
    fn default() -> Self {
        Self {
            kernel_name: "main".into(),
        }
    }
}

impl Debug for GLCompute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("gl_compute")
    }
}

impl SpirvTarget for GLCompute {
    fn set_modes(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        main: Word,
        builtins: Vec<Word>,
        cube_dims: Vec<u32>,
    ) {
        let interface: Vec<u32> = builtins
            .into_iter()
            .chain(iter::once(b.state.params))
            .chain(b.state.shared.values().map(|it| it.id))
            .collect();

        let version = b.compilation_options.vulkan.max_spirv_version;

        b.capability(Capability::Shader);
        b.capability(Capability::PhysicalStorageBufferAddresses);
        b.capability(Capability::VulkanMemoryModel);
        b.capability(Capability::VulkanMemoryModelDeviceScope);
        b.capability(Capability::GroupNonUniform);

        if b.compilation_options.vulkan.supports_explicit_smem {
            b.extension("SPV_KHR_workgroup_memory_explicit_layout");
        }

        if b.compilation_options.vulkan.supports_long_vectors {
            b.extension("SPV_EXT_long_vector");
            b.capability(Capability::LongVectorEXT);
        }

        if b.addr_type.size_bits() == 64 {
            b.extension("SPV_EXT_shader_64bit_indexing");
            b.capability(Capability::Shader64BitIndexingEXT);
            b.execution_mode(main, ExecutionMode::Shader64BitIndexingEXT, []);
        }

        let mut caps = b.capabilities.clone();

        if caps.contains(&Capability::CooperativeMatrixKHR) {
            b.extension("SPV_KHR_cooperative_matrix");
        }

        if caps.contains(&Capability::CooperativeMatrixReductionsNV)
            || caps.contains(&Capability::CooperativeMatrixConversionsNV)
            || caps.contains(&Capability::CooperativeMatrixPerElementOperationsNV)
            || caps.contains(&Capability::CooperativeMatrixTensorAddressingNV)
            || caps.contains(&Capability::CooperativeMatrixBlockLoadsNV)
        {
            b.extension("SPV_NV_cooperative_matrix2")
        }

        // Callback requires physical storage buffer
        if caps.contains(&Capability::CooperativeMatrixBlockLoadsNV) {
            b.extension("SPV_KHR_physical_storage_buffer");
            caps.insert(Capability::PhysicalStorageBufferAddresses);
        }

        if caps.contains(&Capability::TensorAddressingNV) {
            b.extension("SPV_NV_tensor_addressing")
        }

        if caps.contains(&Capability::AtomicFloat16AddEXT) {
            b.extension("SPV_EXT_shader_atomic_float16_add");
        }

        if caps.contains(&Capability::AtomicFloat32AddEXT)
            | caps.contains(&Capability::AtomicFloat64AddEXT)
        {
            b.extension("SPV_EXT_shader_atomic_float_add");
        }

        if caps.contains(&Capability::AtomicFloat16MinMaxEXT)
            | caps.contains(&Capability::AtomicFloat32MinMaxEXT)
            | caps.contains(&Capability::AtomicFloat64MinMaxEXT)
        {
            b.extension("SPV_EXT_shader_atomic_float_min_max");
        }

        if caps.contains(&Capability::AtomicFloat16VectorNV) {
            b.extension("SPV_NV_shader_atomic_fp16_vector");
        }

        if caps.contains(&Capability::BFloat16TypeKHR)
            || caps.contains(&Capability::BFloat16CooperativeMatrixKHR)
            || caps.contains(&Capability::BFloat16DotProductKHR)
        {
            b.extension("SPV_KHR_bfloat16");
        }

        if caps.contains(&Capability::Float8EXT)
            || caps.contains(&Capability::Float8CooperativeMatrixEXT)
        {
            b.extension("SPV_EXT_float8");
        }

        if caps.contains(&Capability::FloatControls2) {
            b.extension("SPV_KHR_float_controls2");
        }

        if b.debug_symbols {
            b.extension("SPV_KHR_non_semantic_info");
        }

        if version < (1, 5) {
            b.extension("SPV_KHR_physical_storage_buffer");
            b.extension("SPV_KHR_vulkan_memory_model");
            if caps.contains(&Capability::StorageBuffer8BitAccess) {
                b.extension("SPV_KHR_8bit_storage");
            }
        }

        if version < (1, 3) && caps.contains(&Capability::StorageBuffer16BitAccess) {
            b.extension("SPV_KHR_16bit_storage");
        }

        for cap in caps {
            b.capability(cap);
        }

        b.memory_model(
            AddressingModel::PhysicalStorageBuffer64,
            MemoryModel::Vulkan,
        );
        b.entry_point(
            ExecutionModel::GLCompute,
            main,
            &self.kernel_name,
            interface,
        );
        b.execution_mode(main, spirv::ExecutionMode::LocalSize, cube_dims);
    }

    fn generate_params(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        bindings: &[KernelArg],
        visibility: &[BufferVisibility],
    ) -> Vec<Buffer> {
        let params_class = Self::params_storage_class(b, bindings.len());

        let params_struct_id = b.id();
        let params_ptr_id = b.id();

        let buffers = bindings
            .iter()
            .map(|binding| self.generate_storage_buffer(b, binding))
            .collect::<Vec<_>>();
        let info = b.info.has_info().then(|| self.generate_info_binding(b));

        b.type_struct_id(
            Some(params_struct_id),
            buffers
                .iter()
                .chain(info.iter())
                .map(|it| it.struct_ptr_ty_id),
        );
        b.type_pointer(Some(params_ptr_id), params_class, params_struct_id);

        b.decorate(params_struct_id, Decoration::Block, []);
        b.name(params_struct_id, "Params");

        let params = b.insert_in_root(|b| b.variable(params_ptr_id, None, params_class, None));
        b.name(params, "params");

        b.state.params = params;

        if !matches!(params_class, StorageClass::PushConstant) {
            b.decorate(params, Decoration::DescriptorSet, vec![0u32.into()]);
            b.decorate(params, Decoration::Binding, vec![0u32.into()]);
        }

        for (i, visibility) in visibility.iter().enumerate() {
            let offset = (size_of::<u64>() * i) as u32;
            b.member_decorate(
                params_struct_id,
                i as u32,
                Decoration::Offset,
                [offset.into()],
            );
            if !visibility.readable {
                b.member_decorate(params_struct_id, i as u32, Decoration::NonReadable, []);
            }
            if !visibility.writable {
                b.member_decorate(params_struct_id, i as u32, Decoration::NonWritable, []);
            }
        }

        if let Some(info) = info {
            let i = buffers.len();
            let offset = (size_of::<u64>() * i) as u32;
            b.member_decorate(
                params_struct_id,
                i as u32,
                Decoration::Offset,
                [offset.into()],
            );
            b.member_decorate(params_struct_id, i as u32, Decoration::NonWritable, []);

            b.state.info = Some(info);
        }

        buffers
    }

    fn load_params(b: &mut SpirvCompiler<Self>) {
        let params = b.state.params;
        let params_class = Self::params_storage_class(b, b.state.buffers.len());

        for (i, buffer) in b.state.buffers.clone().into_iter().enumerate() {
            // uniform/push constant pointer to physical storage buffer pointer
            let field_ptr_ty = b.type_pointer(None, params_class, buffer.struct_ptr_ty_id);
            let field_idx = b.const_u32(i as u32);
            let ptr = b
                .in_bounds_access_chain(field_ptr_ty, None, params, [field_idx])
                .unwrap();
            b.insert_in_setup(|b| {
                b.load(buffer.struct_ptr_ty_id, Some(buffer.id), ptr, None, [])
                    .unwrap()
            });
            b.name(buffer.id, "buffers");
        }

        if let Some(info) = b.state.info {
            let i = b.state.buffers.len();

            // uniform/push constant pointer to physical storage buffer pointer
            let field_ptr_ty = b.type_pointer(None, params_class, info.struct_ptr_ty_id);
            let field_idx = b.const_u32(i as u32);
            let ptr = b
                .in_bounds_access_chain(field_ptr_ty, None, params, [field_idx])
                .unwrap();
            b.insert_in_setup(|b| {
                b.load(info.struct_ptr_ty_id, Some(info.id), ptr, None, [])
                    .unwrap()
            });
            b.name(info.id, "info");
        }
    }

    fn info_storage_class(_b: &mut SpirvCompiler<Self>) -> StorageClass {
        StorageClass::PhysicalStorageBuffer
    }

    fn params_storage_class(b: &mut SpirvCompiler<Self>, num_buffers: usize) -> StorageClass {
        let num_addresses = match b.info.has_info() {
            true => num_buffers + 1,
            false => num_buffers,
        };
        if num_addresses > b.compilation_options.vulkan.push_constant_size / size_of::<u64>() {
            StorageClass::Uniform
        } else {
            StorageClass::PushConstant
        }
    }

    fn set_kernel_name(&mut self, name: impl Into<String>) {
        self.kernel_name = name.into();
    }
}

impl GLCompute {
    fn generate_storage_buffer(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        binding: &KernelArg,
    ) -> Buffer {
        let item = b.compile_type(binding.ty);
        let item_id = item.id(b);
        match item.elem().size() {
            1 => {
                b.capabilities.insert(Capability::StorageBuffer8BitAccess);
            }
            2 => {
                b.capabilities.insert(Capability::StorageBuffer16BitAccess);
            }
            _ => {}
        }

        let ty_size = item.size();

        let arr_ty_id = b.id();
        let struct_ty_id = b.id();
        let storage_class = StorageClass::PhysicalStorageBuffer;

        b.type_runtime_array_id(Some(arr_ty_id), item_id);
        b.decorate(arr_ty_id, Decoration::ArrayStride, [ty_size.into()]);

        b.type_struct_id(Some(struct_ty_id), [arr_ty_id]);
        b.decorate(struct_ty_id, Decoration::Block, []);
        b.member_decorate(struct_ty_id, 0, Decoration::Offset, [0u32.into()]);

        let struct_ptr_ty_id = b.type_pointer(None, storage_class, struct_ty_id);

        Buffer {
            id: b.id(),
            struct_ty_id,
            struct_ptr_ty_id,
            storage_class,
        }
    }

    /// Generate info binding struct and variable.
    /// SPIR-V structs have explicit offsets so unlike other targets we don't need to pad the length.
    fn generate_info_binding(&mut self, b: &mut SpirvCompiler<Self>) -> Buffer {
        let address_type = b.addr_type;
        let struct_ty_id = b.id();
        let storage_class = StorageClass::PhysicalStorageBuffer;

        let mut fields = Vec::new();

        let scalars = b.info.scalars.clone();

        for scalar in scalars {
            let scalar_ty = b.compile_storage_type(scalar.ty);
            match scalar_ty.size() {
                1 => {
                    b.capabilities.insert(Capability::StorageBuffer8BitAccess);
                    b.capabilities
                        .insert(Capability::UniformAndStorageBuffer8BitAccess);
                }
                2 => {
                    b.capabilities.insert(Capability::StorageBuffer16BitAccess);
                    b.capabilities
                        .insert(Capability::UniformAndStorageBuffer16BitAccess);
                }
                _ => {}
            }

            let ty_size = scalar_ty.size();
            let scalar_ty_id = Item::Scalar(scalar_ty).id(b);
            let arr_ty_id = b.id();
            let len_id = b.const_u32(scalar.padded_size() as u32);

            b.type_array_id(Some(arr_ty_id), scalar_ty_id, len_id);
            b.decorate(arr_ty_id, Decoration::ArrayStride, [ty_size.into()]);
            b.name(arr_ty_id, format!("Scalars<{}>", scalar.ty));

            b.member_decorate(
                struct_ty_id,
                fields.len() as u32,
                Decoration::Offset,
                [(scalar.offset as u32).into()],
            );
            fields.push(arr_ty_id);
        }

        if let Some(field) = b.info.sized_meta {
            let scalar_ty = b.compile_storage_type(field.ty);

            let ty_size = scalar_ty.size();
            let scalar_ty_id = Item::Scalar(scalar_ty).id(b);
            let arr_ty_id = b.id();
            let len_id = b.const_u32(field.size as u32);

            b.type_array_id(Some(arr_ty_id), scalar_ty_id, len_id);
            b.decorate(arr_ty_id, Decoration::ArrayStride, [ty_size.into()]);
            b.name(arr_ty_id, "StaticMeta");

            b.member_decorate(
                struct_ty_id,
                fields.len() as u32,
                Decoration::Offset,
                [(field.offset as u32).into()],
            );
            fields.push(arr_ty_id);
        }

        if b.info.has_dynamic_meta {
            let offset = b.info.dynamic_meta_offset;
            let scalar_ty = b.compile_storage_type(address_type);

            let ty_size = scalar_ty.size();
            let scalar_ty_id = Item::Scalar(scalar_ty).id(b);
            let arr_ty_id = b.id();

            b.type_runtime_array_id(Some(arr_ty_id), scalar_ty_id);
            b.decorate(arr_ty_id, Decoration::ArrayStride, [ty_size.into()]);
            b.name(arr_ty_id, "DynamicMeta");

            b.member_decorate(
                struct_ty_id,
                fields.len() as u32,
                Decoration::Offset,
                [Operand::LiteralBit32(offset as u32)],
            );
            fields.push(arr_ty_id);
        }

        b.type_struct_id(Some(struct_ty_id), fields);
        b.decorate(struct_ty_id, Decoration::Block, vec![]);
        b.name(struct_ty_id, "Info");

        let struct_ptr_ty_id = b.type_pointer(None, storage_class, struct_ty_id);

        Buffer {
            id: b.id(),
            struct_ty_id,
            struct_ptr_ty_id,
            storage_class,
        }
    }
}
