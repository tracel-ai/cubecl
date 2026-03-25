use super::{Body, Component, Dialect, Elem, Flags, INFO_NAME, Item, Variable};
use cubecl_core::{
    CubeDim,
    ir::Id,
    prelude::{Location, Visibility},
};

use std::{collections::HashSet, fmt::Display};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct KernelArg<D: Dialect> {
    pub id: Id,
    pub item: Item<D>,
    pub location: Location,
    pub size: Option<usize>,
    pub vis: Visibility,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SharedMemory<D: Dialect> {
    Array {
        index: Id,
        item: Item<D>,
        length: usize,
        align: usize,
        offset: usize,
    },
    Value {
        index: Id,
        item: Item<D>,
        align: usize,
        offset: usize,
    },
}

impl<D: Dialect> SharedMemory<D> {
    pub fn size(&self) -> usize {
        match self {
            SharedMemory::Array { item, length, .. } => *length * item.size(),
            SharedMemory::Value { item, .. } => item.size(),
        }
    }

    pub fn align(&self) -> usize {
        match self {
            SharedMemory::Array { align, .. } => *align,
            SharedMemory::Value { align, .. } => *align,
        }
    }

    pub fn offset(&self) -> usize {
        match self {
            SharedMemory::Array { offset, .. } => *offset,
            SharedMemory::Value { offset, .. } => *offset,
        }
    }
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
    pub size: usize,
}

impl<D: Dialect> LocalArray<D> {
    pub fn new(index: Id, item: Item<D>, size: usize) -> Self {
        Self { index, item, size }
    }
}

impl<D: Dialect> SharedMemory<D> {
    pub fn new_array(index: Id, item: Item<D>, size: usize, align: usize) -> Self {
        Self::Array {
            index,
            item,
            length: size,
            align,
            offset: 0, // initialized later
        }
    }

    pub fn new_value(index: Id, item: Item<D>, align: usize) -> Self {
        Self::Value {
            index,
            item,
            align,
            offset: 0, // initialized later
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComputeKernel<D: Dialect> {
    pub tensor_maps: Vec<KernelArg<D>>,
    pub buffers: Vec<KernelArg<D>>,
    pub scalars: Vec<(Elem<D>, usize)>,
    pub info: cubecl_core::Info,
    pub meta_static_len: usize,
    pub body: Body<D>,
    pub cube_dim: CubeDim,
    pub cluster_dim: Option<CubeDim>,
    pub extensions: Vec<D::Extension>,
    pub flags: Flags<D>,
    pub items: HashSet<super::Item<D>>,
    pub kernel_name: String,
}

impl<D: Dialect> ComputeKernel<D> {
    pub fn shared_memory_size(&self) -> usize {
        let smems = self.body.shared_memories.iter();
        let ends = smems.map(|it| it.offset() + it.size());
        ends.max().unwrap_or_default()
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
        D::compile_type_definitions(f, &self.items, &self.scalars, &self.info, &flags)?;
        D::compile_polyfills(f, &flags)?;
        D::compile_extensions(f, &self.extensions)?;

        // Kernel signature --------------------------------------------------
        D::compile_kernel_signature(
            f,
            &self.kernel_name,
            &self.tensor_maps,
            &self.buffers,
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

pub fn type_info_definition_sized<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    info: &cubecl_core::Info,
    scalars: &[(Elem<D>, usize)],
    address_type: Item<D>,
) -> std::fmt::Result {
    let scalars = info
        .scalars
        .iter()
        .zip(scalars)
        .map(|(field, (ty, _))| format!("{ty} scalars_{ty}[{}];", field.padded_size()))
        .collect::<Vec<_>>()
        .join("\n");
    let static_meta = info
        .sized_meta
        .as_ref()
        .map(|field| format!("{address_type} static_meta[{}];", field.padded_size()))
        .unwrap_or_default();
    write!(
        f,
        "
struct info_st {{
    {scalars}{static_meta}
}};
"
    )
}

pub fn compile_bindings<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    tensor_maps: &[KernelArg<D>],
    buffers: &[KernelArg<D>],
    trailing_comma: bool,
) -> core::fmt::Result {
    write!(f, "    ")?;

    let mut args = Vec::new();

    args.extend(tensor_maps.iter().map(|binding| {
        format!(
            "const __grid_constant__ CUtensorMap tensor_map_{}",
            binding.id
        )
    }));
    args.extend(
        tensor_maps
            .iter()
            .chain(buffers.iter())
            .map(|binding| match binding.vis {
                Visibility::Read => {
                    format!("const {}* __restrict__ buffer_{}", binding.item, binding.id)
                }
                Visibility::ReadWrite => {
                    format!("{}* buffer_{}", binding.item, binding.id)
                }
            }),
    );

    write!(f, "{}", args.join(", "))?;
    if trailing_comma {
        f.write_str(", ")?;
    }
    Ok(())
}

pub fn compile_info_dynamic<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    flags: &Flags<D>,
) -> core::fmt::Result {
    if flags.has_info {
        write!(f, "const info_st* __restrict__ {INFO_NAME}_ptr")
    } else {
        Ok(())
    }
}

pub fn compile_info_static<D: Dialect>(
    f: &mut std::fmt::Formatter<'_>,
    flags: &Flags<D>,
) -> core::fmt::Result {
    let mut inputs = Vec::new();

    if flags.has_dynamic_meta {
        inputs.push(format!(
            "const {}* __restrict__ dynamic_meta",
            flags.address_type
        ))
    }

    if flags.has_info {
        inputs.push(format!("const __grid_constant__ info_st {INFO_NAME}"));
    }

    write!(f, "{}", inputs.join(", "))
}

fn compile_cube_builtin_bindings_decl<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    settings: &Flags<D>,
) -> core::fmt::Result {
    if settings.indexes.absolute_pos_tuple {
        D::compile_absolute_pos_tuple_computation(f)?;
    }

    if settings.indexes.unit_pos {
        D::compile_unit_pos_computation(f)?;
    }

    if settings.indexes.absolute_pos {
        let variable = Variable::<D>::AbsolutePos(settings.address_type.elem);
        let ty = variable.item();
        let absolute_pos_x = Variable::<D>::AbsolutePosX.fmt_cast_to(ty);
        let absolute_pos_y = Variable::<D>::AbsolutePosY.fmt_cast_to(ty);
        let absolute_pos_z = Variable::<D>::AbsolutePosZ.fmt_cast_to(ty);
        let cube_count_x = Variable::<D>::CubeCountX.fmt_cast_to(ty);
        let cube_count_y = Variable::<D>::CubeCountY.fmt_cast_to(ty);
        let cube_dim_x = Variable::<D>::CubeDimX.fmt_cast_to(ty);
        let cube_dim_y = Variable::<D>::CubeDimY.fmt_cast_to(ty);
        writeln!(
            f,
            "{ty} {variable} = (
                {absolute_pos_z} * {cube_count_x} * {cube_dim_x} * {cube_count_y} * {cube_dim_y})
                + ({absolute_pos_y} * {cube_count_x} * {cube_dim_x})
                + {absolute_pos_x};"
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
        let variable = Variable::<D>::CubeCount(settings.address_type.elem);
        let ty = variable.item();
        let cube_count_x = Variable::<D>::CubeCountX.fmt_cast_to(ty);
        let cube_count_y = Variable::<D>::CubeCountY.fmt_cast_to(ty);
        let cube_count_z = Variable::<D>::CubeCountZ.fmt_cast_to(ty);
        writeln!(
            f,
            "{ty} {variable} = {cube_count_x} * {cube_count_y} * {cube_count_z};"
        )?;
    }

    if settings.indexes.cube_pos {
        let variable = Variable::<D>::CubePos(settings.address_type.elem);
        let ty = variable.item();
        let cube_pos_x = Variable::<D>::CubePosX.fmt_cast_to(ty);
        let cube_pos_y = Variable::<D>::CubePosY.fmt_cast_to(ty);
        let cube_pos_z = Variable::<D>::CubePosZ.fmt_cast_to(ty);
        let cube_count_x = Variable::<D>::CubeCountX.fmt_cast_to(ty);
        let cube_count_y = Variable::<D>::CubeCountY.fmt_cast_to(ty);
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
