use super::{Body, Component, Dialect, Elem, Flags, Item, Variable};
use cubecl_core::{
    CubeDim,
    compute::{Location, Visibility},
    ir::Id,
};
use std::{collections::HashSet, fmt::Display};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding<D: Dialect> {
    pub item: Item<D>,
    pub location: Location,
    pub size: Option<usize>,
    pub vis: Visibility,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SharedMemory<D: Dialect> {
    pub index: Id,
    pub item: Item<D>,
    pub size: u32,
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
    pub fn new(index: Id, item: Item<D>, size: u32) -> Self {
        Self { index, item, size }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeKernel<D: Dialect> {
    pub body: Body<D>,
    pub cube_dim: CubeDim,
    pub extensions: Vec<D::Extension>,
    pub flags: Flags,
    pub inputs: Vec<Binding<D>>,
    pub items: HashSet<super::Item<D>>,
    pub kernel_name: String,
    pub named: Vec<(String, Binding<D>)>,
    pub outputs: Vec<Binding<D>>,
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
        // Program Scope -----------------------------------------------------
        D::compile_includes(f, &self.flags)?;
        D::compile_type_definitions(f, &self.items, &self.flags)?;
        D::compile_extensions(f, &self.extensions)?;

        // Kernel signature --------------------------------------------------
        D::compile_kernel_signature(
            f,
            &self.kernel_name,
            &self.inputs,
            &self.outputs,
            &self.named,
            &self.flags,
        )?;

        // Body --------------------------------------------------------------
        f.write_str(" {\n")?;
        compile_cube_builtin_bindings_decl::<D>(f, &self.flags)?;
        write!(f, "{}", self.body)?;
        f.write_str("\n}")?;

        Ok(())
    }
}

pub fn type_definitions<D: Dialect>(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(f, "typedef unsigned char {};", Elem::<D>::U8)?;
    writeln!(f, "typedef unsigned short {};", Elem::<D>::U16)?;
    writeln!(f, "typedef unsigned int {};", Elem::<D>::U32)?;
    writeln!(f, "typedef unsigned long long int {};", Elem::<D>::U64)?;
    writeln!(f, "typedef long long int {};", Elem::<D>::I64)?;
    Ok(())
}

pub fn type_vectorized_definitions<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    items: &HashSet<Item<D>>,
) -> std::fmt::Result {
    for item in items.iter() {
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
    Ok(())
}

pub fn compile_bindings<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    inputs: &[Binding<D>],
    outputs: &[Binding<D>],
    named: &[(String, Binding<D>)],
) -> std::fmt::Result {
    let num_bindings = inputs.len() + outputs.len() + named.len();
    let mut binding_index = 0;
    for (index, binding) in inputs.iter().enumerate() {
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
    for (index, binding) in outputs.iter().enumerate() {
        binding_index += 1;
        write!(f, "{} output_{}[]", binding.item, index)?;
        if binding_index < num_bindings {
            f.write_str(",")?;
        }
    }
    for (name, binding) in named.iter() {
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
    Ok(())
}

fn compile_cube_builtin_bindings_decl<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    settings: &Flags,
) -> core::fmt::Result {
    if settings.var_absolute_pos_global {
        let variable = Variable::<D>::AbsolutePosGlobal;
        let ty = variable.item();
        let absolute_pos_x = Variable::<D>::AbsolutePosX;
        let absolute_pos_y = Variable::<D>::AbsolutePosY;
        let absolute_pos_z = Variable::<D>::AbsolutePosZ;
        let cube_count_x = Variable::<D>::CubeCountX;
        let cube_count_y = Variable::<D>::CubeCountY;
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        writeln!(f, "{ty} {variable} = ({absolute_pos_z} * {cube_count_x} * {cube_dim_x} * {cube_count_y} * {cube_dim_y}) + ({absolute_pos_y} * {cube_count_x} * {cube_dim_x}) + {absolute_pos_x};")?;
    }

    if settings.var_cube_dim_global {
        let variable = Variable::<D>::CubeDimGlobal;
        let ty = variable.item();
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        let cube_dim_z = Variable::<D>::CubeDimZ;
        writeln!(f, "{ty} {variable} = {cube_dim_x} * {cube_dim_y} * {cube_dim_z};")?;
    }

    if settings.var_cube_count_global {
        let variable = Variable::<D>::CubeCountGlobal;
        let ty = variable.item();
        let cube_count_x = Variable::<D>::CubeCountX;
        let cube_count_y = Variable::<D>::CubeCountY;
        let cube_count_z = Variable::<D>::CubeCountZ;
        writeln!(f, "{ty} {variable} = {cube_count_x} * {cube_count_y} * {cube_count_z};")?;
    }

    if settings.var_cube_pos_global {
        let variable = Variable::<D>::CubePosGlobal;
        let ty = variable.item();
        let cube_pos_x = Variable::<D>::CubePosX;
        let cube_pos_y = Variable::<D>::CubePosY;
        let cube_pos_z = Variable::<D>::CubePosZ;
        let cube_count_x = Variable::<D>::CubeCountX;
        let cube_count_y = Variable::<D>::CubeCountY;
        writeln!(f, "{ty} {variable} = ({cube_pos_z} * {cube_count_y} * {cube_count_x}) + ({cube_pos_y} * {cube_count_x}) + {cube_pos_x};")?;
    }

    if settings.var_plane_dim_checked {
        let plane_dim = Variable::<D>::PlaneDim;
        let variable = Variable::<D>::PlaneDimChecked;
        let ty = variable.item();
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        let cube_dim_z = Variable::<D>::CubeDimZ;
        writeln!(f, "{ty} {variable} = min({plane_dim}, {cube_dim_x} * {cube_dim_y} * {cube_dim_z});")?;
    }

    Ok(())
}
