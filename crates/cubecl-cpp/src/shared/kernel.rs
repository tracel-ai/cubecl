use super::{Body, Dialect, Elem, Item, Variable};
use cubecl_core::{CubeDim, compute::Visibility, ir::Id};
use std::{collections::HashSet, fmt::Display};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding<D: Dialect> {
    pub id: Id,
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
    pub tensor_maps: Vec<Id>,
    pub buffers: Vec<Binding<D>>,
    pub scalars: Vec<(Elem<D>, usize)>,
    pub meta_static_len: usize,
    pub cube_dim: CubeDim,
    pub body: Body<D>,
    pub wmma_activated: bool,
    pub pipeline: bool,
    pub barrier: bool,
    pub tma: bool,
    pub bf16: bool,
    pub f16: bool,
    pub grid_constant: bool,
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
        if !self.tensor_maps.is_empty() {
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

        f.write_str("typedef unsigned int uint;\n")?;
        f.write_str("typedef unsigned char uint8;\n")?;
        f.write_str("typedef unsigned short uint16;\n")?;
        f.write_str("typedef unsigned int uint32;\n")?;
        f.write_str("typedef unsigned long long int uint64;\n")?;

        f.write_str("typedef signed char int8;\n")?;
        f.write_str("typedef signed short int16;\n")?;
        f.write_str("typedef signed int int32;\n")?;
        f.write_str("typedef signed long long int int64;\n")?;
        D::deftypes(f)?;

        if self.grid_constant && !self.scalars.is_empty() {
            for (elem, len) in self.scalars.iter() {
                write!(
                    f,
                    "
struct scalars_{elem}_st {{
    {elem} x[{len}];
}};
"
                )?;
            }
        }

        if self.grid_constant && self.meta_static_len > 0 {
            write!(
                f,
                "
struct metadata_st {{
uint x[{}];
}};
",
                self.meta_static_len
            )?;
        }

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

        let meta_len = match (self.grid_constant, self.meta_static_len > 0) {
            (true, true) => 2,
            (false, true) => 1,
            _ => 0,
        };

        let num_bindings =
            self.tensor_maps.len() + self.buffers.len() + self.scalars.len() + meta_len;
        let mut binding_index = 0;
        for index in self.tensor_maps.iter() {
            binding_index += 1;
            write!(f, "const __grid_constant__ CUtensorMap tensormap_{}", index)?;
            if binding_index < num_bindings {
                f.write_str(",")?;
            }
        }
        for binding in self.buffers.iter() {
            binding_index += 1;
            match binding.vis {
                Visibility::Read => {
                    write!(f, "{} buffer_{}[]", binding.item, binding.id)?;
                    // TODO: It breaks slices, because we can't easily create pointer to __restrict__,
                    // we should have multiple pointer types to enable that optimization.
                    //
                    // write!(f, "const {}* __restrict__ input_{}", binding.item, index)?;
                }
                Visibility::ReadWrite => {
                    write!(f, "{} buffer_{}[]", binding.item, binding.id)?;
                }
            }
            if binding_index < num_bindings {
                f.write_str(",")?;
            }
        }
        if self.meta_static_len > 0 {
            binding_index += 1;
            write!(f, "const uint* __restrict__ info")?;
            if binding_index < num_bindings {
                f.write_str(",")?;
            }
        }

        // We use grid constants when supported, since they're much faster than global accesses, and
        // as far as I can tell even faster than normal `__constant__` memory since they're private
        // to the grid and use a special 4KB memory region that exists specifically for kernel
        // parameters.
        if self.grid_constant {
            // Need to sort elements because of alignment when packing
            // Metadata is align 4 so it needs to be spliced in the middle.
            let scalars_of_size = |f: &mut core::fmt::Formatter<'_>,
                                   size: usize,
                                   binding_index: &mut usize|
             -> core::fmt::Result {
                for (elem, _) in self.scalars.iter().filter(|it| it.0.size() == size) {
                    *binding_index += 1;
                    write!(
                        f,
                        "const __grid_constant__ scalars_{elem}_st scalars_{elem}"
                    )?;
                    if *binding_index < num_bindings {
                        f.write_str(",")?;
                    }
                }
                Ok(())
            };

            // Pack 64-bit aligned types first, since metadata is 32-bit aligned
            scalars_of_size(f, 8, &mut binding_index)?;

            // Pack metadata
            if self.meta_static_len > 0 {
                binding_index += 1;
                write!(f, "const __grid_constant__ metadata_st static_info")?;
                if binding_index < num_bindings {
                    f.write_str(",")?;
                }
            }

            // Pack remaining scalars that are 4 bytes or below
            for size in [4, 2, 1] {
                scalars_of_size(f, size, &mut binding_index)?;
            }
        } else {
            for (elem, _) in self.scalars.iter() {
                binding_index += 1;
                write!(f, "const __restrict__ {}* scalars_{}", elem, elem)?;

                if binding_index < num_bindings {
                    f.write_str(",")?;
                }
            }
        }

        f.write_str("\n) {\n")?;

        write!(f, "{}", self.body)?;
        f.write_str("\n}")?;

        Ok(())
    }
}
