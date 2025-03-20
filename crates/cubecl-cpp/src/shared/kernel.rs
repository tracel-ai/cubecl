use super::{Body, Dialect, Item, Variable};
use cubecl_core::{
    CubeDim,
    compute::{ConstBinding, Visibility},
    ir::Id,
};
use std::{collections::HashSet, fmt::Display};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding<D: Dialect> {
    pub item: Item<D>,
    pub size: Option<usize>,
    pub vis: Visibility,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedMemory<D: Dialect> {
    pub index: Id,
    pub item: Item<D>,
    pub size: u32,
    pub align: Option<u32>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ConstArray<D: Dialect> {
    pub index: Id,
    pub item: Item<D>,
    pub size: u32,
    pub values: Vec<Variable<D>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LocalArray<D: Dialect> {
    pub index: Id,
    pub item: Item<D>,
    pub size: u32,
}

impl<D: Dialect> LocalArray<D> {
    pub fn new(index: Id, item: Item<D>, size: u32) -> Self {
        Self { index, item, size }
    }
}

impl<D: Dialect> SharedMemory<D> {
    pub fn new(index: Id, item: Item<D>, size: u32, align: Option<u32>) -> Self {
        Self {
            index,
            item,
            size,
            align,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeKernel<D: Dialect> {
    pub constants: Vec<ConstBinding>,
    pub inputs: Vec<Binding<D>>,
    pub outputs: Vec<Binding<D>>,
    pub named: Vec<(String, Binding<D>)>,
    pub cube_dim: CubeDim,
    pub body: Body<D>,
    pub wmma_activated: bool,
    pub pipeline: bool,
    pub barrier: bool,
    pub tma: bool,
    pub bf16: bool,
    pub f16: bool,
    pub fast_math: bool,
    pub items: HashSet<super::Item<D>>,
    pub kernel_name: String,
}

impl<D: Dialect> ComputeKernel<D> {
    pub fn shared_memory_size(&self) -> usize {
        let mut current = 0usize;

        for var in self.body.shared_memories.iter() {
            let factor = var.item.vectorization;
            let elem_size_bytes = var.item.elem().size();
            current += (var.size as usize) * factor * elem_size_bytes;
        }

        current
    }
}

impl<D: Dialect> Display for ComputeKernel<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut tma = self.tma;
        if self
            .constants
            .iter()
            .any(|c| matches!(c, ConstBinding::TensorMap))
        {
            tma = true;
        }

        if self.bf16 {
            D::include_bf16(f)?;
        }

        if self.f16 {
            D::include_f16(f)?;
        }

        if self.wmma_activated {
            D::wmma_includes(f)?;
        }

        if self.pipeline {
            f.write_str("#include <cooperative_groups/memcpy_async.h>\n")?;
            f.write_str("#include <cuda/pipeline>\n")?;
        }
        if self.barrier || tma {
            f.write_str("#include <cooperative_groups.h>\n")?;
            f.write_str("#include <cooperative_groups/memcpy_async.h>\n")?;
            f.write_str("#include <cuda/barrier>\n")?;
        }
        if tma {
            f.write_str(
                "typedef struct CUtensorMap_st {
alignas(64) unsigned long long int opaque[16];
} CUtensorMap;\n",
            )?;
        }

        f.write_str("typedef unsigned char uint8;\n")?;
        f.write_str("typedef unsigned short uint16;\n")?;
        f.write_str("typedef unsigned int uint;\n")?;
        f.write_str("typedef unsigned long long int uint64;\n")?;
        f.write_str("typedef long long int int64;\n")?;
        D::deftypes(f)?;

        for item in self.items.iter() {
            let elem = item.elem;
            let size = item.vectorization;
            let alignment = elem.size() * size;
            if size > 1 {
                write!(
                    f,
                    "
struct __align__({alignment}) {item} {{"
                )?;

                for i in 0..size {
                    write!(
                        f,
                        "
    {elem} i_{i};"
                    )?;
                }

                f.write_str("\n};\n")?;
            }
        }

        write!(
            f,
            "

extern \"C\" __global__ void {}(
",
            self.kernel_name
        )?;

        let num_bindings =
            self.constants.len() + self.inputs.len() + self.outputs.len() + self.named.len();
        let mut binding_index = 0;
        for (index, binding) in self.constants.iter().enumerate() {
            binding_index += 1;
            match binding {
                ConstBinding::TensorMap => {
                    write!(f, "const __grid_constant__ CUtensorMap constant_{}", index)?;
                }
            }
            if binding_index < num_bindings {
                f.write_str(",")?;
            }
        }
        for (index, binding) in self.inputs.iter().enumerate() {
            binding_index += 1;
            match binding.vis {
                Visibility::Read => {
                    write!(f, "{} input_{}[]", binding.item, index)?;
                    // TODO: It breaks slices, because we can't easily create pointer to __restrict__,
                    // we should have multiple pointer types to enable that optimization.
                    //
                    // write!(f, "const {}* __restrict__ input_{}", binding.item, index)?;
                }
                Visibility::ReadWrite => {
                    write!(f, "{} input_{}[]", binding.item, index)?;
                }
            }
            if binding_index < num_bindings {
                f.write_str(",")?;
            }
        }
        for (index, binding) in self.outputs.iter().enumerate() {
            binding_index += 1;
            write!(f, "{} output_{}[]", binding.item, index)?;
            if binding_index < num_bindings {
                f.write_str(",")?;
            }
        }
        for (name, binding) in self.named.iter() {
            binding_index += 1;

            match binding.vis {
                Visibility::Read => {
                    write!(f, "{} {}[]", binding.item, name)?;
                    // TODO: It breaks slices, because we can't easily create pointer to __restrict__,
                    // we should have multiple pointer types to enable that optimization.
                    //
                    // write!(f, "const {}* __restrict__ {}", binding.item, name)?;
                }
                Visibility::ReadWrite => {
                    write!(f, "{} {}[]", binding.item, name)?;
                }
            }

            if binding_index < num_bindings {
                f.write_str(",")?;
            }
        }

        f.write_str("\n) {\n")?;

        write!(f, "{}", self.body)?;
        f.write_str("\n}")?;

        Ok(())
    }
}
