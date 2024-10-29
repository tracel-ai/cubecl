use super::{Body, Dialect, Item, Variable};
use cubecl_core::{ir::CubeDim, CompilerRepresentation};
use std::{collections::HashSet, fmt::Display};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding<D: Dialect> {
    pub item: Item<D>,
    pub size: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedMemory<D: Dialect> {
    pub index: u16,
    pub item: Item<D>,
    pub size: u32,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ConstArray<D: Dialect> {
    pub index: u16,
    pub item: Item<D>,
    pub size: u32,
    pub values: Vec<Variable<D>>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LocalArray<D: Dialect> {
    pub index: u16,
    pub item: Item<D>,
    pub depth: u8,
    pub size: u32,
}

impl<D: Dialect> LocalArray<D> {
    pub fn new(index: u16, item: Item<D>, depth: u8, size: u32) -> Self {
        Self {
            index,
            item,
            depth,
            size,
        }
    }
}

impl<D: Dialect> SharedMemory<D> {
    pub fn new(index: u16, item: Item<D>, size: u32) -> Self {
        Self { index, item, size }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeKernel<D: Dialect> {
    pub inputs: Vec<Binding<D>>,
    pub outputs: Vec<Binding<D>>,
    pub named: Vec<(String, Binding<D>)>,
    pub cube_dim: CubeDim,
    pub body: Body<D>,
    pub wmma_activated: bool,
    pub bf16: bool,
    pub f16: bool,
    pub items: HashSet<super::Item<D>>,
}

impl<D: Dialect> CompilerRepresentation for ComputeKernel<D> {
    fn shared_memory_size(&self) -> usize {
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
        if self.wmma_activated {
            D::include_wmma(f)?;
        }
        if self.bf16 {
            D::include_bf16(f)?;
        }

        if self.f16 {
            D::include_f16(f)?;
        }

        f.write_str("typedef unsigned int uint;\n")?;

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

extern \"C\" __global__ void kernel(
",
        )?;

        let num_bindings = self.inputs.len() + self.outputs.len() + self.named.len();
        let mut binding_index = 0;
        for (index, binding) in self.inputs.iter().enumerate() {
            binding_index += 1;
            write!(f, "{} input_{}[]", binding.item, index)?;
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
            write!(f, "{} {}[]", binding.item, name)?;

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
