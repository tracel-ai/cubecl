use cubecl_core::compute::{Binding, Location, Visibility};
use hashbrown::HashMap;
use rspirv::spirv::{
    self, AddressingModel, Capability, Decoration, ExecutionModel, MemoryModel, StorageClass, Word,
};
use std::fmt::Debug;

use crate::{
    SpirvCompiler, debug,
    extensions::{TargetExtensions, glcompute},
    item::Item,
};

pub trait SpirvTarget:
    TargetExtensions<Self> + Debug + Clone + Default + Send + Sync + 'static
{
    fn extensions(&mut self, b: &mut SpirvCompiler<Self>) -> HashMap<String, Word>;
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
        index: u32,
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
            .chain(b.state.inputs.iter().copied())
            .chain(b.state.outputs.iter().copied())
            .chain(b.state.named.values().copied())
            .chain(b.state.shared_memories.values().map(|it| it.id))
            .collect();

        b.capability(Capability::Shader);
        b.capability(Capability::VulkanMemoryModel);
        b.capability(Capability::VulkanMemoryModelDeviceScope);

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

        if caps.contains(&Capability::FloatControls2) {
            b.declare_float_execution_modes(main);
        }
    }

    fn generate_binding(
        &mut self,
        b: &mut SpirvCompiler<Self>,
        binding: Binding,
        name: String,
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

    fn extensions(&mut self, b: &mut SpirvCompiler<Self>) -> HashMap<String, Word> {
        let mut extensions = HashMap::new();
        extensions.insert(
            glcompute::STD_NAME.to_string(),
            b.ext_inst_import(glcompute::STD_NAME),
        );
        if b.debug_symbols {
            extensions.insert(
                debug::DEBUG_EXT_NAME.to_string(),
                b.ext_inst_import(debug::DEBUG_EXT_NAME),
            );
            extensions.insert(
                debug::PRINT_EXT_NAME.to_string(),
                b.ext_inst_import(debug::PRINT_EXT_NAME),
            );
        }
        extensions
    }

    fn set_kernel_name(&mut self, name: impl Into<String>) {
        self.kernel_name = name.into();
    }
}
