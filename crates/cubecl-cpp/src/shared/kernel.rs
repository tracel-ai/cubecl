use crate::shared::STATIC_INFO_NAME;

use super::{Body, Component, Dialect, Elem, Flags, INFO_NAME, Item, Variable};
use cubecl_core::{
    CubeDim,
    compute::{Location, Visibility},
    ir::Id,
};

use std::{collections::HashSet, fmt::Display};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding<D: Dialect> {
    pub id: Id,
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
    pub align: Option<u32>,
    pub offset: u32,
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
            offset: 0, // initialized later
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeKernel<D: Dialect> {
    pub tensor_maps: Vec<Id>,
    pub buffers: Vec<Binding<D>>,
    pub scalars: Vec<(Elem<D>, usize)>,
    pub meta_static_len: usize,
    pub body: Body<D>,
    pub cube_dim: CubeDim,
    pub cluster_dim: Option<CubeDim>,
    pub extensions: Vec<D::Extension>,
    pub flags: Flags,
    pub items: HashSet<super::Item<D>>,
    pub kernel_name: String,
}

impl<D: Dialect> ComputeKernel<D> {
    pub fn shared_memory_size(&self) -> usize {
        // Account for alignment padding between shared memory buffers
        // Sorted to minimize that padding
        let mut shared_memories = self.body.shared_memories.clone();
        shared_memories.sort_by_key(|smem| smem.align.unwrap_or(smem.item.size() as u32));
        shared_memories.reverse();

        let mut current = 0usize;

        for var in self.body.shared_memories.iter() {
            let align = var.align.unwrap_or(var.item.size() as u32);
            let size_bytes = var.size as usize * var.item.size();
            let offset = current.next_multiple_of(align as usize);
            current = offset + size_bytes;
        }

        current
    }
}

impl<D: Dialect> Display for ComputeKernel<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut flags = self.flags.clone();
        if !self.tensor_maps.is_empty() {
            flags.inst_tma = true;
        }

        // Program Scope -----------------------------------------------------
        D::compile_includes(f, &flags)?;
        D::compile_type_definitions(f, &self.items, &self.scalars, &flags)?;
        D::compile_polyfills(f, &flags)?;
        D::compile_extensions(f, &self.extensions)?;

        // Kernel signature --------------------------------------------------
        D::compile_kernel_signature(
            f,
            &self.kernel_name,
            &self.tensor_maps,
            &self.buffers,
            &self.scalars,
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
    writeln!(f, "typedef unsigned int uint;")?;
    writeln!(f, "typedef unsigned char uint8;")?;
    writeln!(f, "typedef unsigned short uint16;")?;
    writeln!(f, "typedef unsigned int uint32;")?;
    writeln!(f, "typedef unsigned long long int uint64;")?;

    writeln!(f, "typedef signed char int8;")?;
    writeln!(f, "typedef signed short int16;")?;
    writeln!(f, "typedef signed int int32;")?;
    writeln!(f, "typedef signed long long int int64;")?;

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

            f.write_str("\n};")?;
        }
    }
    Ok(())
}

pub fn type_scalar_definitions<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    scalars: &[(Elem<D>, usize)],
) -> std::fmt::Result {
    for (elem, count) in scalars.iter() {
        writeln!(
            f,
            "
struct scalars_{elem}_st {{
{elem} x[{count}];
}};"
        )?;
    }
    Ok(())
}

pub fn type_info_definition<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    static_len: usize,
) -> std::fmt::Result {
    if static_len > 0 {
        write!(
            f,
            "
struct metadata_st {{
uint x[{static_len}];
}};
"
        )?;
    }
    Ok(())
}

pub fn compile_bindings<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    tensor_maps: &[Id],
    buffers: &[Binding<D>],
    trailing_comma: bool,
    flags: &Flags,
) -> core::fmt::Result {
    write!(f, "    ")?;

    let mut args = Vec::new();

    args.extend(
        tensor_maps
            .iter()
            .map(|id| format!("const __grid_constant__ CUtensorMap tensor_map_{id}")),
    );
    args.extend(buffers.iter().map(|binding| match binding.vis {
        Visibility::Read => {
            format!("const {}* __restrict__ buffer_{}", binding.item, binding.id)
        }
        Visibility::ReadWrite => {
            format!("{}* buffer_{}", binding.item, binding.id)
        }
    }));
    args.extend(
        flags
            .has_dynamic_meta
            .then(|| format!("const uint32* __restrict__ {INFO_NAME}")),
    );

    write!(f, "{}", args.join(", "))?;
    if trailing_comma {
        f.write_str(", ")?;
    }
    Ok(())
}

pub fn compile_scalars_dynamic<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    scalars: &[(Elem<D>, usize)],
) -> core::fmt::Result {
    let scalar_inputs = scalars
        .iter()
        .map(|(elem, _)| format!("const {elem}* __restrict__ scalars_{elem}"));
    let scalar_inputs = scalar_inputs.collect::<Vec<String>>();

    write!(f, "{}", scalar_inputs.join(","))
}

pub fn compile_scalars_static<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    scalars: &[(Elem<D>, usize)],
    flags: &Flags,
) -> core::fmt::Result {
    let mut scalar_inputs = Vec::new();

    // Need to sort elements because of alignment when packing
    // Metadata is align 4 so it needs to be spliced in the middle.
    let scalars_of_size = |scalar_inputs: &mut Vec<String>, size: usize| {
        for (elem, _) in scalars.iter().filter(|it| it.0.size() == size) {
            scalar_inputs.push(format!(
                "const __grid_constant__ scalars_{elem}_st scalars_{elem}"
            ));
        }
    };

    // Pack 64-bit aligned types first, since metadata is 32-bit aligned
    scalars_of_size(&mut scalar_inputs, 8);

    // Pack metadata
    if flags.static_meta_length > 0 {
        scalar_inputs.push(format!(
            "const __grid_constant__ metadata_st {STATIC_INFO_NAME}"
        ));
    }

    // Pack remaining scalars that are 4 bytes or below
    for size in [4, 2, 1] {
        scalars_of_size(&mut scalar_inputs, size);
    }

    write!(f, "{}", scalar_inputs.join(", "))
}

fn compile_cube_builtin_bindings_decl<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    settings: &Flags,
) -> core::fmt::Result {
    if settings.indexes.absolute_pos_tuple {
        D::compile_absolute_pos_tuple_computation(f)?;
    }

    if settings.indexes.unit_pos {
        D::compile_unit_pos_computation(f)?;
    }

    if settings.indexes.absolute_pos {
        let variable = Variable::<D>::AbsolutePos;
        let ty = variable.item();
        let absolute_pos_x = Variable::<D>::AbsolutePosX;
        let absolute_pos_y = Variable::<D>::AbsolutePosY;
        let absolute_pos_z = Variable::<D>::AbsolutePosZ;
        let cube_count_x = Variable::<D>::CubeCountX;
        let cube_count_y = Variable::<D>::CubeCountY;
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        writeln!(
            f,
            "{ty} {variable} = ({absolute_pos_z} * {cube_count_x} * {cube_dim_x} * {cube_count_y} * {cube_dim_y}) + ({absolute_pos_y} * {cube_count_x} * {cube_dim_x}) + {absolute_pos_x};"
        )?;
    }

    if settings.indexes.cube_dim {
        let variable = Variable::<D>::CubeDim;
        let ty = variable.item();
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        let cube_dim_z = Variable::<D>::CubeDimZ;
        writeln!(
            f,
            "{ty} {variable} = {cube_dim_x} * {cube_dim_y} * {cube_dim_z};"
        )?;
    }

    if settings.indexes.cube_count {
        let variable = Variable::<D>::CubeCount;
        let ty = variable.item();
        let cube_count_x = Variable::<D>::CubeCountX;
        let cube_count_y = Variable::<D>::CubeCountY;
        let cube_count_z = Variable::<D>::CubeCountZ;
        writeln!(
            f,
            "{ty} {variable} = {cube_count_x} * {cube_count_y} * {cube_count_z};"
        )?;
    }

    if settings.indexes.cube_pos {
        let variable = Variable::<D>::CubePos;
        let ty = variable.item();
        let cube_pos_x = Variable::<D>::CubePosX;
        let cube_pos_y = Variable::<D>::CubePosY;
        let cube_pos_z = Variable::<D>::CubePosZ;
        let cube_count_x = Variable::<D>::CubeCountX;
        let cube_count_y = Variable::<D>::CubeCountY;
        writeln!(
            f,
            "{ty} {variable} = ({cube_pos_z} * {cube_count_y} * {cube_count_x}) + ({cube_pos_y} * {cube_count_x}) + {cube_pos_x};"
        )?;
    }

    if settings.indexes.plane_dim_checked {
        let plane_dim = Variable::<D>::PlaneDim;
        let variable = Variable::<D>::PlaneDimChecked;
        let ty = variable.item();
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        let cube_dim_z = Variable::<D>::CubeDimZ;
        writeln!(
            f,
            "{ty} {variable} = min({plane_dim}, {cube_dim_x} * {cube_dim_y} * {cube_dim_z});"
        )?;
    }

    if settings.indexes.cluster_pos {
        f.write_str(
            "
cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
",
        )?;
    }

    Ok(())
}
