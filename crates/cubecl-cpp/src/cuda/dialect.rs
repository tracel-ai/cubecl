use std::collections::HashSet;

use cubecl_core::ir::Id;

use crate::{
    Dialect,
    cuda::ptx::TMA_LOAD_IM2COL,
    shared::{
        self, Binding, Component, DialectBindings, DialectCubeBuiltins, DialectIncludes,
        DialectInstructions, DialectTypes, DialectWmmaCompiler, Elem, Flags, Instruction, Item,
        SharedMemory, Variable, WarpInstruction, unary,
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
        if flags.op_barrier || flags.inst_tma || flags.indexes.cluster_pos {
            f.write_str("#include <cooperative_groups.h>\n")?;
            f.write_str("#include <cooperative_groups/memcpy_async.h>\n")?;
            f.write_str("#include <cuda/barrier>\n")?;
        }
        if flags.inst_tma {
            f.write_str(
                "typedef struct CUtensorMap_st {
alignas(64) unsigned long long int opaque[16];
} CUtensorMap;\n",
            )?;
        }
        Ok(())
    }

    fn compile_extensions(
        _f: &mut std::fmt::Formatter<'_>,
        _extensions: &[Self::Extension],
    ) -> std::fmt::Result {
        Ok(())
    }

    fn register_instruction_extension(
        _extensions: &mut Vec<Self::Extension>,
        _instruction: &Instruction<Self>,
    ) {
    }

    fn register_warp_instruction_extension(
        _extensions: &mut Vec<Self::Extension>,
        _instruction: &WarpInstruction<Self>,
    ) {
    }
}

// Types

impl DialectTypes<Self> for CudaDialect {
    fn item_can_be_optimized() -> bool {
        true
    }

    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &HashSet<Item<Self>>,
        scalars: &[(Elem<Self>, usize)],
        flags: &Flags,
    ) -> std::fmt::Result {
        shared::type_definitions::<Self>(f)?;
        shared::type_vectorized_definitions::<Self>(f, items)?;
        if flags.use_grid_constants {
            shared::type_scalar_definitions::<Self>(f, scalars)?;
            shared::type_info_definition::<Self>(f, flags.static_meta_length)?;
        }
        Self::compile_wmma_type_definitions(f)?;
        Ok(())
    }

    fn compile_polyfills(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result {
        if flags.inst_tma_im2col {
            writeln!(f, "{TMA_LOAD_IM2COL}")?;
        }
        Ok(())
    }

    fn compile_elem(
        f: &mut std::fmt::Formatter<'_>,
        elem: &shared::Elem<Self>,
        words: bool,
    ) -> std::fmt::Result {
        if words {
            match elem {
                shared::Elem::F32 => f.write_str("float"),
                shared::Elem::F64 => f.write_str("double"),
                shared::Elem::TF32 => f.write_str("float"),
                shared::Elem::I8 => f.write_str("char"),
                shared::Elem::I16 => f.write_str("short"),
                shared::Elem::I32 => f.write_str("int"),
                shared::Elem::I64 => f.write_str("long"),
                shared::Elem::U8 => f.write_str("uchar"),
                shared::Elem::U16 => f.write_str("ushort"),
                shared::Elem::U32 => f.write_str("uint"),
                shared::Elem::U64 => f.write_str("ulong"),
                _ => Self::compile_elem(f, elem, false),
            }
        } else {
            match elem {
                shared::Elem::F16 => f.write_str("__half"),
                shared::Elem::F162 => f.write_str("__half2"),
                shared::Elem::F32 => f.write_str("float"),
                shared::Elem::F64 => f.write_str("double"),
                shared::Elem::BF16 => f.write_str("__nv_bfloat16"),
                shared::Elem::BF162 => f.write_str("__nv_bfloat162"),
                shared::Elem::TF32 => f.write_str("float"),
                shared::Elem::I8 => f.write_str("int8"),
                shared::Elem::I16 => f.write_str("int16"),
                shared::Elem::I32 => f.write_str("int32"),
                shared::Elem::I64 => f.write_str("int64"),
                shared::Elem::U8 => f.write_str("uint8"),
                shared::Elem::U16 => f.write_str("uint16"),
                shared::Elem::U32 => f.write_str("uint32"),
                shared::Elem::U64 => f.write_str("uint64"),
                shared::Elem::Bool => f.write_str("bool"),
                shared::Elem::Atomic(inner) => write!(f, "{inner}"),
                shared::Elem::_Dialect(_) => Ok(()),
            }
        }
    }

    fn compile_item(f: &mut std::fmt::Formatter<'_>, item: &Item<Self>) -> std::fmt::Result {
        if 1 == item.vectorization {
            return write!(f, "{}", item.elem);
        }
        if item.native {
            // native types use the word form of types only
            Self::compile_elem(f, &item.elem, true)?;
            write!(f, "{}", item.vectorization)
        } else {
            write!(f, "{}_{}", item.elem, item.vectorization)
        }
    }

    fn compile_local_memory_qualifier(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_shared_memory_qualifier(
        f: &mut std::fmt::Formatter<'_>,
        shared: &SharedMemory<Self>,
    ) -> std::fmt::Result {
        let align = match shared.align {
            Some(alignment) => format!("alignas({alignment})"),
            None => "".to_string(),
        };
        write!(f, "__shared__ {align}")
    }
}

// Kernel argument bindings

impl DialectBindings<Self> for CudaDialect {
    fn compile_kernel_signature(
        f: &mut std::fmt::Formatter<'_>,
        kernel_name: &str,
        tensor_maps: &[Id],
        buffers: &[Binding<Self>],
        scalars: &[(Elem<Self>, usize)],
        flags: &Flags,
    ) -> std::fmt::Result {
        write!(
            f,
            "

extern \"C\" __global__ void "
        )?;
        if let Some(cluster_dim) = flags.cluster_dim {
            write!(
                f,
                "__cluster_dims__({}, {}, {}) ",
                cluster_dim.x, cluster_dim.y, cluster_dim.z
            )?;
        }
        writeln!(f, "{} (", kernel_name)?;
        let has_scalars =
            !scalars.is_empty() || (flags.use_grid_constants && flags.static_meta_length > 0);
        shared::compile_bindings(f, tensor_maps, buffers, has_scalars, flags)?;
        if flags.use_grid_constants {
            shared::compile_scalars_static(f, scalars, flags)?;
        } else {
            shared::compile_scalars_dynamic(f, scalars)?;
        }
        f.write_str("\n)")?;
        //
        Ok(())
    }
}

// Cube builtins dialect

impl DialectCubeBuiltins<Self> for CudaDialect {
    fn compile_cluster_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cluster.block_rank()")
    }

    fn compile_cluster_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cluster.block_index().x")
    }

    fn compile_cluster_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cluster.block_index().y")
    }

    fn compile_cluster_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cluster.block_index().z")
    }
}

// Instructions

impl DialectInstructions<Self> for CudaDialect {
    // sync
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "__syncthreads();\n")
    }

    fn compile_instruction_thread_fence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "__threadfence();")
    }

    // unary
    fn compile_instruction_find_first_set<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match input.elem() {
            Elem::I32 => write!(f, "__ffs({input})"),
            Elem::U32 => write!(f, "__ffs({}({input}))", Elem::<Self>::I32),
            Elem::I64 => write!(f, "__ffsll({input})"),
            Elem::U64 => write!(f, "__ffsll({}({input}))", Elem::<Self>::I64),
            _ => write!(f, "__ffs({}({input}))", Elem::<Self>::I32),
        }?;
        write!(f, ")")
    }

    fn compile_instruction_leading_zeros_scalar<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match input.elem() {
            Elem::I32 => write!(f, "__clz({input})"),
            Elem::U32 => write!(f, "__clz({}({input}))", Elem::<Self>::I32),
            Elem::I64 => write!(f, "__clzll({input})"),
            Elem::U64 => write!(f, "__clzll({}({input}))", Elem::<Self>::I64),
            in_elem => write!(
                f,
                "{out_elem}(__clz({}) - {})",
                unary::zero_extend(input),
                (size_of::<u32>() - in_elem.size()) * 8
            ),
        }?;
        write!(f, ")")
    }

    // others
    fn compile_instruction_max_function_name(
        f: &mut std::fmt::Formatter<'_>,
        item: Item<Self>,
    ) -> std::fmt::Result {
        let max = match item.elem() {
            Elem::F16 | Elem::BF16 => "__hmax",
            Elem::F162 | Elem::BF162 => "__hmax2",
            _ => "max",
        };
        write!(f, "{max}")
    }

    fn compile_instruction_min_function_name(
        f: &mut std::fmt::Formatter<'_>,
        item: Item<Self>,
    ) -> std::fmt::Result {
        let min = match item.elem() {
            Elem::F16 | Elem::BF16 => "__hmin",
            Elem::F162 | Elem::BF162 => "__hmin2",
            _ => "min",
        };
        write!(f, "{min}")
    }

    // warp
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
        _elem: &Elem<Self>,
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
    fn compile_warp_all<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result {
        write!(f, "__all_sync(-1, {input})")
    }
    fn compile_warp_any<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result {
        write!(f, "__any_sync(-1, {input})")
    }

    fn compile_warp_ballot(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<Self>,
        _out_elem: &Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "__ballot_sync(-1, {input})")
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
