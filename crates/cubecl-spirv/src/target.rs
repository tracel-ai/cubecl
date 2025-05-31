use cubecl_core::compute::{Binding, Location, Visibility};
use rspirv::{
    dr,
    spirv::{
        self, AddressingModel, Capability, Decoration, ExecutionModel, MemoryModel, StorageClass,
        Word,
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
        binding: Binding,
        name: String,
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
            .chain(b.state.scalar_bindings.values().copied())
            .chain(b.state.shared_memories.values().map(|it| it.id))
            .collect();

        b.capability(Capability::Shader);
        b.capability(Capability::VulkanMemoryModel);
        b.capability(Capability::VulkanMemoryModelDeviceScope);

        let caps: Vec<_> = b.capabilities.iter().copied().collect();
        for cap in caps.iter() {
            b.capability(*cap);
        }
        if b.float_controls {
            let inst = dr::Instruction::new(
                spirv::Op::Capability,
                None,
                None,
                vec![dr::Operand::LiteralBit32(6029)],
            );
            b.module_mut().capabilities.push(inst);
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

        if b.float_controls {
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

        if b.float_controls {
            b.declare_float_execution_modes(main);
        }
    }

    fn generate_binding(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        binding: Binding,
        name: String,
    ) -> Word {
        let index = binding.id;
        let item = b.compile_item(binding.item);
        let item = match binding.size {
            Some(size) => Item::Array(Box::new(item), size as u32),
            None => Item::RuntimeArray(Box::new(item)),
        };
        let arr = item.id(b); // pre-generate type
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

    fn set_kernel_name(&mut self, name: impl Into<String>) {
        self.kernel_name = name.into();
    }
}
