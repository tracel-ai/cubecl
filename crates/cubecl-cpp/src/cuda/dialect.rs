use std::collections::HashSet;
use std::fmt::Display;

use crate::{
    Dialect,
    shared::{
        self, Binding, DialectBindings, DialectCubeBuiltins, DialectIncludes, DialectTypes,
        DialectWarp, DialectWmmaCompiler, Flags, Instruction, Item,
    },
};

use super::{Extension, arch::CudaArchitecture, mma::CudaWmmaCompiler};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CudaDialect {}

// Base dialect

impl Dialect for CudaDialect {}

// Includes

impl DialectIncludes<Self> for CudaDialect {
    type Extension = Extension;

    fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result {
        f.write_str("#include <cuda_runtime.h>\n")?;
        if flags.elem_bf16 {
            f.write_str("#include <cuda_bf16.h>\n")?;
        }
        if flags.elem_f16 {
            f.write_str("#include <cuda_fp16.h>\n")?;
        }
        if flags.inst_wmma {
            Self::compile_wmma_includes(f)?;
        }
        if flags.op_pipeline {
            f.write_str("#include <cooperative_groups/memcpy_async.h>\n")?;
            f.write_str("#include <cuda/pipeline>\n")?;
        }
        if flags.op_barrier {
            f.write_str("#include <cooperative_groups.h>\n")?;
            f.write_str("#include <cooperative_groups/memcpy_async.h>\n")?;
            f.write_str("#include <cuda/barrier>\n")?;
        }
        Ok(())
    }

    fn compile_extensions(
        _f: &mut std::fmt::Formatter<'_>,
        _extensions: &[Self::Extension],
    ) -> std::fmt::Result {
        Ok(())
    }

    fn register_extension(
        _extensions: &mut Vec<Self::Extension>,
        _instruction: &Instruction<Self>,
    ) {
    }
}

// Types

impl DialectTypes<Self> for CudaDialect {
    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &HashSet<Item<Self>>,
        _flags: &Flags,
    ) -> std::fmt::Result {
        shared::type_definitions::<Self>(f)?;
        shared::type_vectorized_definitions::<Self>(f, items)?;
        Self::compile_wmma_type_definitions(f)?;
        Ok(())
    }

    fn compile_elem(
        f: &mut std::fmt::Formatter<'_>,
        elem: &shared::Elem<Self>,
    ) -> std::fmt::Result {
        match elem {
            shared::Elem::F16 => f.write_str("__half"),
            shared::Elem::F162 => f.write_str("__half2"),
            shared::Elem::F32 => f.write_str("float"),
            shared::Elem::F64 => f.write_str("double"),
            shared::Elem::BF16 => f.write_str("__nv_bfloat16"),
            shared::Elem::BF162 => f.write_str("__nv_bfloat162"),
            shared::Elem::TF32 => f.write_str("float"),
            shared::Elem::I8 => f.write_str("char"),
            shared::Elem::I16 => f.write_str("short"),
            shared::Elem::I32 => f.write_str("int"),
            shared::Elem::I64 => f.write_str("int64"),
            shared::Elem::U8 => f.write_str("uint8"),
            shared::Elem::U16 => f.write_str("uint16"),
            shared::Elem::U32 => f.write_str("uint"),
            shared::Elem::U64 => f.write_str("uint64"),
            shared::Elem::Bool => f.write_str("bool"),
            shared::Elem::Atomic(inner) => inner.fmt(f),
            shared::Elem::_Dialect(_) => Ok(()),
        }
    }

    fn compile_item(f: &mut std::fmt::Formatter<'_>, item: &Item<Self>) -> std::fmt::Result {
        if 1 == item.vectorization {
            return write!(f, "{}", item.elem);
        }
        write!(f, "{}_{}", item.elem, item.vectorization)
    }

    fn compile_local_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_shared_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "__shared__")
    }
}

// Kernel argument bindings

impl DialectBindings<Self> for CudaDialect {
    fn compile_kernel_signature(
        f: &mut std::fmt::Formatter<'_>,
        kernel_name: &str,
        inputs: &[Binding<Self>],
        outputs: &[Binding<Self>],
        named: &[(String, Binding<Self>)],
        _flags: &Flags,
    ) -> std::fmt::Result {
        write!(
            f,
            "

extern \"C\" __global__ void {}(
",
            kernel_name
        )?;
        shared::compile_bindings::<Self>(f, inputs, outputs, named)?;
        f.write_str("\n)")
    }
}

// Cube builtins dialect

impl DialectCubeBuiltins for CudaDialect {
    fn compile_absolute_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("absoluteIdx")
    }

    fn compile_absolute_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("idxGlobal")
    }

    fn compile_absolute_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos(f)?;
        write!(f, ".x")
    }

    fn compile_absolute_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos(f)?;
        write!(f, ".y")
    }

    fn compile_absolute_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos(f)?;
        write!(f, ".z")
    }

    fn compile_cube_count(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("gridDim")
    }

    fn compile_cube_count_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("gridDimGlobal")
    }

    fn compile_cube_count_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count(f)?;
        write!(f, ".x")
    }

    fn compile_cube_count_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count(f)?;
        write!(f, ".y")
    }

    fn compile_cube_count_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count(f)?;
        write!(f, ".z")
    }

    fn compile_cube_dim(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockDim")
    }

    fn compile_cube_dim_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockDimGlobal")
    }

    fn compile_cube_dim_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim(f)?;
        write!(f, ".x")
    }

    fn compile_cube_dim_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim(f)?;
        write!(f, ".y")
    }

    fn compile_cube_dim_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim(f)?;
        write!(f, ".z")
    }

    fn compile_cube_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockIdx")
    }

    fn compile_cube_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockIdxGlobal")
    }

    fn compile_cube_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos(f)?;
        write!(f, ".x")
    }

    fn compile_cube_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos(f)?;
        write!(f, ".y")
    }

    fn compile_cube_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos(f)?;
        write!(f, ".z")
    }

    fn compile_unit_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threadIdxGlobal")
    }

    fn compile_unit_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threadIdx")
    }

    fn compile_unit_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos(f)?;
        write!(f, ".x")
    }

    fn compile_unit_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos(f)?;
        write!(f, ".y")
    }

    fn compile_unit_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos(f)?;
        write!(f, ".z")
    }
}

// Warp

impl DialectWarp for CudaDialect {
    fn compile_warp_shuffle(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        source: &str,
    ) -> std::fmt::Result {
        write!(f, "__shfl_sync(-1, {var}, {source})")
    }
    fn compile_warp_shuffle_xor(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "__shfl_xor_sync(-1, {var}, {offset})")
    }
    fn compile_warp_shuffle_up(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "__shfl_up_sync(-1, {var}, {offset})")
    }
    fn compile_warp_shuffle_down(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "__shfl_down_sync(-1, {var}, {offset})")
    }
    fn compile_warp_all(f: &mut std::fmt::Formatter<'_>, var: &str) -> std::fmt::Result {
        write!(f, "__all_sync(-1, {var})")
    }
    fn compile_warp_any(f: &mut std::fmt::Formatter<'_>, var: &str) -> std::fmt::Result {
        write!(f, "__any_sync(-1, {var})")
    }
    fn compile_warp_ballot(f: &mut std::fmt::Formatter<'_>, out: &str) -> std::fmt::Result {
        write!(f, "__ballot_sync(-1, {out})")
    }
}

// Coop Matrices dialect

impl DialectWmmaCompiler<Self> for CudaDialect {
    type Architecture = CudaArchitecture;

    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        CudaWmmaCompiler::compile_wmma_includes(f)
    }

    fn compile_wmma_type_definitions(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        CudaWmmaCompiler::compile_wmma_type_definitions(f)
    }

    fn compile_local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        CudaWmmaCompiler::compile_local_variables(f)
    }

    fn compile_fragment_ident(
        ident: &crate::shared::FragmentIdent<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_fragment_ident(ident, f)
    }

    fn compile_fragment_layout(
        layout: &crate::shared::FragmentLayout<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_fragment_layout(layout, f)
    }

    fn compile_fragment(
        fragment: &crate::shared::Fragment<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_fragment(fragment, f)
    }

    fn compile_instruction(
        instruction: &crate::shared::WmmaInstruction<Self>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        CudaWmmaCompiler::compile_instruction(instruction, f)
    }

    fn supported_wmma_combinations(
        arch: &Self::Architecture,
    ) -> crate::shared::SupportedWmmaCombinations {
        CudaWmmaCompiler::supported_wmma_combinations(arch)
    }
}
