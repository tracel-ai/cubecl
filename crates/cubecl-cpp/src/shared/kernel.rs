use crate::{
    shared::{OpToCPP, ty::TypeExtCPP},
    target::{CtxTarget, Target},
};

use cubecl_core::ir::metadata::Info;
use cubecl_opt::analyses::liveness::shared::SmemAllocation;
use pliron::{builtin::ops::ModuleOp, context::Context, dict_key};

use core::fmt::{Display, Write};

dict_key!(ATTR_RESTRICT, "restrict");
dict_key!(ATTR_CONST, "const");

pub struct ComputeKernel {
    pub ctx: Context,
    pub module: ModuleOp,
    pub shared_memories: Vec<SmemAllocation>,
    pub shared_memory_size: usize,
    pub info_by_ptr: bool,
    pub has_dynamic_meta: bool,
}

impl Display for ComputeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let module = self.module.to_cpp(&self.ctx);
        f.write_str(&module)
    }
}

impl ComputeKernel {
    pub fn shared_memory_size(&self, ctx: &Context) -> usize {
        let smems = self.shared_memories.iter();
        let ends = smems.map(|it| it.offset + it.smem.size(ctx));
        ends.max().unwrap_or_default()
    }
}

// impl<D: Dialect> Display for ComputeKernel<D> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let mut flags = self.flags.clone();
//         if !self.tensor_maps.is_empty() {
//             flags.inst_tma = true;
//         }

//         // Program Scope -----------------------------------------------------
//         D::compile_includes(f, &flags)?;
//         D::compile_type_definitions(f, &self.items, &self.scalars, &self.info, &flags)?;
//         D::compile_polyfills(f, &flags)?;
//         D::compile_extensions(f, &self.extensions)?;

//         // Kernel signature --------------------------------------------------
//         D::compile_kernel_signature(
//             f,
//             &self.kernel_name,
//             &self.tensor_maps,
//             &self.buffers,
//             &self.flags,
//         )?;

//         // Body --------------------------------------------------------------
//         f.write_str(" {\n")?;
//         compile_cube_builtin_bindings_decl(f, &self.flags)?;
//         write!(f, "{}", self.body)?;
//         f.write_str("\n}")?;

//         Ok(())
//     }
// }

pub fn type_definitions(f: &mut dyn Write, ctx: &Context) -> std::fmt::Result {
    writeln!(f, "typedef unsigned int uint32_t;")?;
    writeln!(f, "typedef unsigned char uint8_t;")?;
    writeln!(f, "typedef unsigned short uint16_t;")?;
    writeln!(f, "typedef unsigned long long int uint64_t;")?;

    writeln!(f, "typedef signed char int8_t;")?;
    writeln!(f, "typedef signed short int16_t;")?;
    writeln!(f, "typedef signed int int32_t;")?;
    writeln!(f, "typedef signed long long int int64_t;")?;

    if ctx.target() != Target::Metal {
        define_array_polyfill(f)?;
    }

    Ok(())
}

/// Define a minimal version of C++'s `std::array` so we can match Rust semantics on arrays.
pub fn define_array_polyfill(f: &mut dyn Write) -> core::fmt::Result {
    writeln!(
        f,
        "
template <typename T, size_t N>
struct array {{
    T data[N];
    __device__ T& operator[](size_t i) {{ return data[i]; }}
    __device__ const T& operator[](size_t i) const {{ return data[i]; }}
}};"
    )
}

// pub fn type_vectorized_definitions(
//     f: &mut std::fmt::Formatter<'_>,
//     items: &HashSet<TypeHandle>,
// ) -> std::fmt::Result {
//     for item in items.iter().filter(|it| it.vectorization() > 1) {
//         let elem = item.elem();
//         let size = item.vectorization();
//         let alignment = elem.size() * size;
//         if size > 1 {
//             write!(
//                 f,
//                 "
// struct __align__({alignment}) {item} {{"
//             )?;

//             for i in 0..size {
//                 write!(
//                     f,
//                     "
//     {elem} i_{i};"
//                 )?;
//             }

//             f.write_str("\n};")?;
//         }
//     }
//     Ok(())
// }

pub fn type_info_definition_sized(
    f: &mut dyn Write,
    ctx: &Context,
    info: &Info,
) -> std::fmt::Result {
    let scalars = info
        .scalars
        .iter()
        .map(|field| {
            let ty = field.ty.to_type(ctx).to_cpp(ctx);
            format!("{ty} scalars_{ty}[{}];", field.padded_size())
        })
        .collect::<Vec<_>>()
        .join("\n");
    let static_meta = info
        .sized_meta
        .as_ref()
        .map(|field| {
            format!(
                "{} static_meta[{}];",
                field.ty.to_type(ctx).to_cpp(ctx),
                field.padded_size()
            )
        })
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

// pub fn compile_info_dynamic(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> core::fmt::Result {
//     if flags.has_info {
//         write!(f, "const info_st* __restrict__ {INFO_NAME}_ptr")
//     } else {
//         Ok(())
//     }
// }

// pub fn compile_info_static(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> core::fmt::Result {
//     let mut inputs = Vec::new();

//     if flags.has_dynamic_meta {
//         inputs.push(format!(
//             "const {}* __restrict__ dynamic_meta",
//             flags.address_type
//         ))
//     }

//     if flags.has_info {
//         inputs.push(format!("const __grid_constant__ info_st {INFO_NAME}"));
//     }

//     write!(f, "{}", inputs.join(", "))
// }

// fn compile_cube_builtin_bindings_decl(
//     f: &mut core::fmt::Formatter<'_>,
//     settings: &Flags,
// ) -> core::fmt::Result {
//     if settings.indexes.absolute_pos_tuple {
//         D::compile_absolute_pos_tuple_computation(f)?;
//     }

//     if settings.indexes.unit_pos {
//         D::compile_unit_pos_computation(f)?;
//     }

//     if settings.indexes.absolute_pos {
//         let value = Builtin::<D>::AbsolutePos(*settings.address_type.elem());
//         let ty = value.item();
//         let absolute_pos_x = Builtin::<D>::AbsolutePosX.fmt_cast_to(ty);
//         let absolute_pos_y = Builtin::<D>::AbsolutePosY.fmt_cast_to(ty);
//         let absolute_pos_z = Builtin::<D>::AbsolutePosZ.fmt_cast_to(ty);
//         let cube_count_x = Builtin::<D>::CubeCountX.fmt_cast_to(ty);
//         let cube_count_y = Builtin::<D>::CubeCountY.fmt_cast_to(ty);
//         let cube_dim_x = Builtin::<D>::CubeDimX.fmt_cast_to(ty);
//         let cube_dim_y = Builtin::<D>::CubeDimY.fmt_cast_to(ty);
//         writeln!(
//             f,
//             "{ty} {value} = (
//                 {absolute_pos_z} * {cube_count_x} * {cube_dim_x} * {cube_count_y} * {cube_dim_y})
//                 + ({absolute_pos_y} * {cube_count_x} * {cube_dim_x})
//                 + {absolute_pos_x};"
//         )?;
//     }

//     if settings.indexes.cube_dim {
//         let value = Builtin::<D>::CubeDim;
//         let ty = value.item();
//         let cube_dim_x = Builtin::<D>::CubeDimX;
//         let cube_dim_y = Builtin::<D>::CubeDimY;
//         let cube_dim_z = Builtin::<D>::CubeDimZ;
//         writeln!(
//             f,
//             "{ty} {value} = {cube_dim_x} * {cube_dim_y} * {cube_dim_z};"
//         )?;
//     }

//     if settings.indexes.cube_count {
//         let value = Builtin::<D>::CubeCount(*settings.address_type.elem());
//         let ty = value.item();
//         let cube_count_x = Builtin::<D>::CubeCountX.fmt_cast_to(ty);
//         let cube_count_y = Builtin::<D>::CubeCountY.fmt_cast_to(ty);
//         let cube_count_z = Builtin::<D>::CubeCountZ.fmt_cast_to(ty);
//         writeln!(
//             f,
//             "{ty} {value} = {cube_count_x} * {cube_count_y} * {cube_count_z};"
//         )?;
//     }

//     if settings.indexes.cube_pos {
//         let value = Builtin::<D>::CubePos(*settings.address_type.elem());
//         let ty = value.item();
//         let cube_pos_x = Builtin::<D>::CubePosX.fmt_cast_to(ty);
//         let cube_pos_y = Builtin::<D>::CubePosY.fmt_cast_to(ty);
//         let cube_pos_z = Builtin::<D>::CubePosZ.fmt_cast_to(ty);
//         let cube_count_x = Builtin::<D>::CubeCountX.fmt_cast_to(ty);
//         let cube_count_y = Builtin::<D>::CubeCountY.fmt_cast_to(ty);
//         writeln!(
//             f,
//             "{ty} {value} = ({cube_pos_z} * {cube_count_y} * {cube_count_x}) + ({cube_pos_y} * {cube_count_x}) + {cube_pos_x};"
//         )?;
//     }

//     if settings.thread_block {
//         f.write_str(
//             "
// cooperative_groups::thread_block thread_block = cooperative_groups::this_thread_block();
// ",
//         )?;
//     }

//     if settings.indexes.cluster_pos {
//         f.write_str(
//             "
// cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
// ",
//         )?;
//     }

//     Ok(())
// }
