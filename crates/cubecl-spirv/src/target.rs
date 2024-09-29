use cubecl_core::ir::{Binding, Location, Visibility};
use rspirv::spirv::{
    self, AddressingModel, Capability, Decoration, ExecutionModel, MemoryModel, StorageClass, Word,
};
use std::fmt::Debug;

use crate::{item::Item, SpirvCompiler};

pub trait SpirvTarget: Debug + Clone + Default + Send + Sync + 'static {
    fn extensions(&mut self, b: &mut SpirvCompiler<Self>) -> Vec<Word>;
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
        index: u32,
    ) -> Word;
}

#[derive(Clone, Default)]
pub struct GLCompute;

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
            .chain(b.state.inputs.iter().copied())
            .chain(b.state.outputs.iter().copied())
            .chain(b.state.named.values().copied())
            .collect();

        b.capability(Capability::Shader);
        b.capability(Capability::VulkanMemoryModel);

        let caps: Vec<_> = b.capabilities.iter().copied().collect();
        for cap in caps {
            b.capability(cap);
        }

        b.memory_model(AddressingModel::Logical, MemoryModel::Vulkan);
        b.entry_point(ExecutionModel::GLCompute, main, "main", interface);
        b.execution_mode(main, spirv::ExecutionMode::LocalSize, cube_dims);
    }

    fn generate_binding(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        binding: Binding,
        index: u32,
    ) -> Word {
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

        if matches!(binding.visibility, Visibility::Read) {
            b.decorate(var, Decoration::NonWritable, vec![]);
        }

        b.decorate(var, Decoration::DescriptorSet, vec![0u32.into()]);
        b.decorate(var, Decoration::Binding, vec![index.into()]);
        b.decorate(struct_ty, Decoration::Block, vec![]);
        b.member_decorate(struct_ty, 0, Decoration::Offset, vec![0u32.into()]);

        var
    }

    fn extensions(&mut self, b: &mut SpirvCompiler<Self>) -> Vec<Word> {
        vec![b.ext_inst_import("GLSL.std.450")]
    }
}
