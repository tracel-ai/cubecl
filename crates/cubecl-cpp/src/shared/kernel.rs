use crate::shared::ty::TypeExtCPP;

use cubecl_core::ir::metadata::Info;
use cubecl_runtime::kernel::Visibility;
use pliron::context::Context;

use core::fmt::{Display, Write};

pub struct ComputeKernel {
    pub ctx: Context,
    pub shared_memory_size: usize,
    pub buffers: Vec<Visibility>,
    /// The emitted source, rendered once during `compile_ir` where emission errors can still
    /// fail the compilation.
    pub source: String,
}

impl Display for ComputeKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.source)
    }
}

pub fn type_definitions(f: &mut dyn Write, long: &str) -> std::fmt::Result {
    writeln!(f, "typedef unsigned int uint32_t;")?;
    writeln!(f, "typedef unsigned char uint8_t;")?;
    writeln!(f, "typedef unsigned short uint16_t;")?;
    writeln!(f, "typedef unsigned {long} int uint64_t;")?;

    writeln!(f, "typedef signed char int8_t;")?;
    writeln!(f, "typedef signed short int16_t;")?;
    writeln!(f, "typedef signed int int32_t;")?;
    writeln!(f, "typedef signed {long} int int64_t;")?;

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
}};\n"
    )
}

pub fn define_tensormap_opaque(f: &mut dyn Write) -> core::fmt::Result {
    f.write_str(
        "
typedef struct CUtensorMap_st {
alignas(128) unsigned long long int opaque[16];
} CUtensorMap;\n",
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
