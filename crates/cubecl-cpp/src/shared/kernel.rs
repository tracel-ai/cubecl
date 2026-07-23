use crate::{
    shared::{OpToCPP, ty::TypeExtCPP},
    target::{CtxTarget, Target},
};

use cubecl_core::ir::{ContextExt, GlobalState, metadata::Info};
use pliron::context::Context;

use core::fmt::{Display, Write};

pub struct ComputeKernel {
    pub ctx: Context,
    pub shared_memory_size: usize,
}

impl Display for ComputeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let module = self.ctx.aux_ty::<GlobalState>().module;
        let module = module.to_cpp(&self.ctx);
        f.write_str(&module)
    }
}

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

    // This is fine to generate even on old cards, it's just an opaque block of memory
    // The headers are dumb and only work in NVCC so we just need to define it ourselves
    if ctx.target() == Target::Cuda {
        define_tensormap_opaque(f)?;
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

pub fn define_tensormap_opaque(f: &mut dyn Write) -> core::fmt::Result {
    f.write_str(
        "
typedef struct CUtensorMap_st {
alignas(128) unsigned long long int opaque[16];
} CUtensorMap;",
    )
}

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
            format!("{ty} scalars_{}[{}];", field.ty, field.padded_size(ctx))
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
                field.padded_size(ctx)
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
