use cubecl_core::{
    ir::StorageType,
    prelude::{KernelArg, Location, ScalarKernelArg, Visibility},
};
use rspirv::{
    dr::Operand,
    spirv::{
        self, AddressingModel, Capability, Decoration, ExecutionMode, ExecutionModel, MemoryModel,
        StorageClass, Word,
    },
};
use std::{fmt::Debug, iter};

use crate::{SpirvCompiler, extensions::TargetExtensions, item::Item};

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
    fn generate_binding(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        binding: KernelArg,
        name: String,
    ) -> Word;
    fn generate_info_binding(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        offset: u32,
        scalars: &[ScalarKernelArg],
        address_type: StorageType,
        static_meta_len: u32,
        has_dynamic_meta: bool,
    ) -> Word;

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
            .chain(b.state.buffers.iter().copied())
            .chain(iter::once(b.state.info))
            .chain(b.state.shared_arrays.values().map(|it| it.id))
            .chain(b.state.shared.values().map(|it| it.id))
            .collect();

        b.capability(Capability::Shader);
        b.capability(Capability::VulkanMemoryModel);
        b.capability(Capability::VulkanMemoryModelDeviceScope);
        b.capability(Capability::GroupNonUniform);

        if b.compilation_options.supports_explicit_smem {
            b.extension("SPV_KHR_workgroup_memory_explicit_layout");
        }

        if b.addr_type.size_bits() == 64 {
            b.extension("SPV_EXT_shader_64bit_indexing");
            b.capability(Capability::Shader64BitIndexingEXT);
            b.execution_mode(main, ExecutionMode::Shader64BitIndexingEXT, []);
        }

        let caps: Vec<_> = b.capabilities.iter().copied().collect();
        for cap in caps.iter() {
            b.capability(*cap);
        }

        if caps.contains(&Capability::CooperativeMatrixKHR) {
            b.extension("SPV_KHR_cooperative_matrix");
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

        b.memory_model(AddressingModel::Logical, MemoryModel::Vulkan);
        b.entry_point(
            ExecutionModel::GLCompute,
            main,
            &self.kernel_name,
            interface,
        );
        b.execution_mode(main, spirv::ExecutionMode::LocalSize, cube_dims);
    }

    fn generate_binding(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        binding: KernelArg,
        name: String,
    ) -> Word {
        let index = binding.id;
        let item = b.compile_type(binding.ty);
        let item_size = item.size();
        let item = match binding.size {
            Some(size) => Item::Array(Box::new(item), size as u32),
            None => Item::RuntimeArray(Box::new(item)),
        };
        let arr = item.id(b); // pre-generate type

        if !b.state.array_types.contains(&arr) {
            b.decorate(arr, Decoration::ArrayStride, [item_size.into()]);
            b.state.array_types.insert(arr);
        }

        let struct_ty = b.id();
        b.type_struct_id(Some(struct_ty), vec![arr]);

        let location = match binding.location {
            Location::Cube => StorageClass::Workgroup,
            Location::Storage => StorageClass::StorageBuffer,
        };
        let ptr_ty = b.type_pointer(None, location, struct_ty);
        let var = b.variable(ptr_ty, None, location, None);

        b.debug_name(var, name);

        if matches!(binding.visibility, Visibility::Read) {
            b.decorate(var, Decoration::NonWritable, vec![]);
        }

        b.decorate(var, Decoration::DescriptorSet, vec![0u32.into()]);
        b.decorate(var, Decoration::Binding, vec![index.into()]);
        b.decorate(struct_ty, Decoration::Block, vec![]);
        b.member_decorate(struct_ty, 0, Decoration::Offset, vec![0u32.into()]);

        var
    }

    /// Generate info binding struct and variable.
    /// SPIR-V structs have explicit offsets so unlike other targets we don't need to pad the length.
    fn generate_info_binding(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        index: u32,
        scalars: &[ScalarKernelArg],
        address_type: StorageType,
        static_meta_len: u32,
        has_dynamic_meta: bool,
    ) -> Word {
        let struct_ty = b.id();

        let mut offset = 0;
        let mut fields = Vec::new();

        for scalar in scalars {
            let scalar_ty = b.compile_storage_type(scalar.ty);
            let arr_ty = Item::Array(Box::new(Item::Scalar(scalar_ty)), scalar.count as u32);
            let arr_ty_id = arr_ty.id(b);

            if !b.state.array_types.contains(&arr_ty_id) {
                b.decorate(
                    arr_ty_id,
                    Decoration::ArrayStride,
                    [(scalar.ty.size() as u32).into()],
                );
                b.state.array_types.insert(arr_ty_id);
            }

            b.member_decorate(
                struct_ty,
                fields.len() as u32,
                Decoration::Offset,
                [(offset as u32).into()],
            );
            fields.push(arr_ty_id);
            offset += (scalar.ty.size() * scalar.count).next_multiple_of(size_of::<u64>());
        }

        if static_meta_len > 0 {
            let scalar_ty = b.compile_storage_type(address_type);
            let arr_ty = Item::Array(Box::new(Item::Scalar(scalar_ty)), static_meta_len);
            let arr_ty_id = arr_ty.id(b);

            if !b.state.array_types.contains(&arr_ty_id) {
                b.decorate(
                    arr_ty_id,
                    Decoration::ArrayStride,
                    [(address_type.size() as u32).into()],
                );
                b.state.array_types.insert(arr_ty_id);
            }

            b.member_decorate(
                struct_ty,
                fields.len() as u32,
                Decoration::Offset,
                [(offset as u32).into()],
            );
            fields.push(arr_ty_id);
            offset +=
                (address_type.size() * static_meta_len as usize).next_multiple_of(size_of::<u64>());
        }

        if has_dynamic_meta {
            let scalar_ty = b.compile_storage_type(address_type);
            let arr_ty = Item::RuntimeArray(Box::new(Item::Scalar(scalar_ty)));
            let arr_ty_id = arr_ty.id(b);

            if !b.state.array_types.contains(&arr_ty_id) {
                b.decorate(
                    arr_ty_id,
                    Decoration::ArrayStride,
                    [(address_type.size() as u32).into()],
                );
                b.state.array_types.insert(arr_ty_id);
            }

            b.member_decorate(
                struct_ty,
                fields.len() as u32,
                Decoration::Offset,
                [Operand::LiteralBit32(offset as u32)],
            );
            fields.push(arr_ty_id);
        }

        b.type_struct_id(Some(struct_ty), fields);

        let location = StorageClass::StorageBuffer;
        let ptr_ty = b.type_pointer(None, location, struct_ty);
        let var = b.variable(ptr_ty, None, location, None);

        b.debug_name(var, "info");
        b.decorate(var, Decoration::NonWritable, vec![]);

        b.decorate(var, Decoration::DescriptorSet, vec![0u32.into()]);
        b.decorate(var, Decoration::Binding, vec![index.into()]);
        b.decorate(struct_ty, Decoration::Block, vec![]);

        var
    }

    fn set_kernel_name(&mut self, name: impl Into<String>) {
        self.kernel_name = name.into();
    }
}
