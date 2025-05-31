use std::fmt::Display;
use std::{collections::HashSet, marker::PhantomData};

use cubecl_core::ir::Id;

use crate::shared::{
    Component, DialectInstructions, Elem, Instruction, SharedMemory, Variable, unary,
};
use crate::{
    Dialect,
    shared::{
        self, Binding, DialectBindings, DialectCubeBuiltins, DialectIncludes, DialectTypes,
        DialectWmmaCompiler, Flags, Item,
    },
};

use super::Extension;
use super::arch::AMDArchitecture;
use super::extension::{format_f162bf16, format_max, format_min};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HipDialect<M> {
    _wmma_compiler: PhantomData<M>,
}

// Base dialect

impl<M: DialectWmmaCompiler<Self>> Dialect for HipDialect<M> {
    type Architecture = AMDArchitecture;
}

// Includes

impl<M: DialectWmmaCompiler<Self>> DialectIncludes<Self> for HipDialect<M> {
    type Extension = Extension<Self>;

    fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result {
        f.write_str("#include <hip/hip_runtime.h>\n")?;
        if flags.elem_bf16 {
            f.write_str("#include <hip/hip_bf16.h>\n")?;
        }
        if flags.elem_f16 {
            f.write_str("#include <hip/hip_fp16.h>\n")?;
        }
        if flags.inst_wmma {
            Self::compile_wmma_includes(f)?;
        }
        Ok(())
    }

    fn compile_extensions(
        f: &mut std::fmt::Formatter<'_>,
        extensions: &[Self::Extension],
    ) -> std::fmt::Result {
        for extension in extensions {
            match extension {
                Extension::F162BF16 => format_f162bf16(f)?,
                Extension::Max(var) => format_max::<Self>(f, var)?,
                Extension::Min(var) => format_min::<Self>(f, var)?,
                Extension::NoExtension => {}
            }
        }
        Ok(())
    }

    fn register_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &Instruction<Self>,
    ) {
        let mut register_extension = |extension: Self::Extension| {
            if !extensions.contains(&extension) {
                extensions.push(extension);
            }
        };
        #[allow(clippy::single_match)]
        match instruction {
            shared::Instruction::<Self>::Max(op) => {
                register_extension(Extension::Max(*op.lhs.item().elem()));
            }
            shared::Instruction::<Self>::Min(op) => {
                register_extension(Extension::Min(*op.lhs.item().elem()));
            }
            _ => {}
        }
    }

    fn register_warp_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &shared::WarpInstruction<Self>,
    ) {
        let mut register_extension = |extension: Self::Extension| {
            if !extensions.contains(&extension) {
                extensions.push(extension);
            }
        };
        #[allow(clippy::single_match)]
        match instruction {
            shared::WarpInstruction::<Self>::ReduceMax { input, .. } => {
                let input_item = input.item();
                let input_elem = input_item.elem();
                if *input_elem == Elem::<Self>::BF16 {
                    register_extension(Extension::F162BF16);
                }
                register_extension(Extension::Max(*input_elem));
            }
            shared::WarpInstruction::<Self>::ReduceMin { input, .. } => {
                let input_item = input.item();
                let input_elem = input_item.elem();
                if *input_elem == Elem::<Self>::BF16 {
                    register_extension(Extension::F162BF16);
                }
                register_extension(Extension::Min(*input_elem));
            }
            shared::WarpInstruction::<Self>::ReduceProd { input, .. } => {
                let input_item = input.item();
                let input_elem = input_item.elem();
                if *input_elem == Elem::<Self>::BF16 {
                    register_extension(Extension::F162BF16);
                }
            }
            shared::WarpInstruction::<Self>::ReduceSum { input, .. } => {
                let input_item = input.item();
                let input_elem = input_item.elem();
                if *input_elem == Elem::<Self>::BF16 {
                    register_extension(Extension::F162BF16);
                }
            }
            _ => {}
        }
    }
}

// Types

impl<M: DialectWmmaCompiler<Self>> DialectTypes<Self> for HipDialect<M> {
    fn item_can_be_optimized() -> bool {
        // for now deactivate support for half2 and bfloat162 because the HIP API lack support for it.
        false
    }

    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &HashSet<Item<Self>>,
        _scalars: &[(Elem<Self>, usize)],
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
                shared::Elem::FP4(_)
                | shared::Elem::FP4x2(_)
                | shared::Elem::FP6(_)
                | shared::Elem::FP6x2(_)
                | shared::Elem::FP8(_)
                | shared::Elem::FP8x2(_) => unimplemented!("FP4/FP6/FP8 not supported in HIP"),
                shared::Elem::F16 => f.write_str("__half"),
                shared::Elem::F16x2 => f.write_str("__half2"),
                shared::Elem::F32 => f.write_str("float"),
                shared::Elem::F64 => f.write_str("double"),
                shared::Elem::BF16 => f.write_str("__bf16"),
                shared::Elem::BF16x2 => f.write_str("__bf162"),
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
                shared::Elem::Atomic(inner) => inner.fmt(f),
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

    fn compile_shared_memory_declaration(
        f: &mut std::fmt::Formatter<'_>,
        shared: &SharedMemory<Self>,
    ) -> std::fmt::Result {
        let item = shared.item;
        let index = shared.index;
        let size = shared.size;
        let alignment = shared
            .align
            .map(|align| format!("alignas({align})"))
            .unwrap_or_default();
        writeln!(
            f,
            "__shared__ {alignment} {item} shared_memory_{index}[{size}];",
        )
    }
}

// Kernel argument bindings

impl<M: DialectWmmaCompiler<Self>> DialectBindings<Self> for HipDialect<M> {
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

extern \"C\" __global__ void {kernel_name}(
"
        )?;
        shared::compile_bindings::<Self>(f, tensor_maps, buffers, !scalars.is_empty(), flags)?;
        shared::compile_scalars_dynamic::<Self>(f, scalars)?;
        f.write_str("\n)")
    }
}

// Cube builtins dialect

impl<M: DialectWmmaCompiler<Self>> DialectCubeBuiltins<Self> for HipDialect<M> {}

// Instructions

impl<M: DialectWmmaCompiler<Self>> DialectInstructions<Self> for HipDialect<M> {
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "__syncthreads();\n")
    }

    fn compile_instruction_sync_warp(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        panic!("Sync warp is unimplemented on hip")
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
            Elem::I32 | Elem::U32 => write!(f, "__ffs({input})"),
            Elem::I64 | Elem::U64 => write!(f, "__ffsll({input})"),
            _ => write!(f, "__ffs({}({input}))", Elem::<Self>::U32),
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
            Elem::I32 | Elem::U32 => write!(f, "__clz({input})"),
            Elem::I64 | Elem::U64 => write!(f, "__clzll({input})"),
            in_elem => write!(
                f,
                "__clz({}) - {}",
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
            Elem::F16 => "__hmax",
            Elem::BF16 => "max_bfloat16",
            _ => "max",
        };
        write!(f, "{max}")
    }

    fn compile_instruction_min_function_name(
        f: &mut std::fmt::Formatter<'_>,
        item: Item<Self>,
    ) -> std::fmt::Result {
        let min = match item.elem() {
            Elem::F16 => "__hmin",
            Elem::BF16 => "min_bfloat16",
            _ => "min",
        };
        write!(f, "{min}")
    }

    // Warp
    fn compile_warp_shuffle(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        source: &str,
    ) -> std::fmt::Result {
        write!(f, "__shfl({var}, {source})")
    }
    fn compile_warp_shuffle_xor(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        elem: &Elem<Self>,
        offset: &str,
    ) -> std::fmt::Result {
        match elem {
            Elem::BF16 => write!(
                f,
                "half_to_bfloat16(__shfl_xor(reinterpret_cast<__half&>({var}), {offset}))"
            ),
            _ => write!(f, "__shfl_xor({var}, {offset})"),
        }
    }
    fn compile_warp_shuffle_up(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "__shfl_up({var}, {offset})")
    }
    fn compile_warp_shuffle_down(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "__shfl_down({var}, {offset})")
    }
    fn compile_warp_all<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result {
        let item = input.item();
        let elem = item.elem;
        write!(f, "static_cast<{elem}>(__all({input}))")
    }
    fn compile_warp_any<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result {
        let item = input.item();
        let elem = item.elem;
        write!(f, "static_cast<{elem}>(__any({input}))")
    }
    fn compile_warp_ballot(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<Self>,
        out_elem: &Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(__ballot({input}))")
    }
}

// Coop Matrices dialect

impl<M: DialectWmmaCompiler<Self>> DialectWmmaCompiler<Self> for HipDialect<M> {
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        M::compile_wmma_includes(f)
    }

    fn compile_wmma_type_definitions(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        M::compile_wmma_type_definitions(f)
    }

    fn compile_wmma_local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        M::compile_wmma_local_variables(f)
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &Variable<Self>,
    ) -> std::fmt::Result {
        M::compile_wmma_fragment_declaration(f, var)
    }

    fn compile_wwma_fragment_ident(
        f: &mut std::fmt::Formatter<'_>,
        ident: &crate::shared::FragmentIdent<Self>,
    ) -> std::fmt::Result {
        M::compile_wwma_fragment_ident(f, ident)
    }

    fn compile_wmma_fragment_layout(
        f: &mut std::fmt::Formatter<'_>,
        layout: &crate::shared::FragmentLayout<Self>,
    ) -> std::fmt::Result {
        M::compile_wmma_fragment_layout(f, layout)
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &crate::shared::Fragment<Self>,
    ) -> std::fmt::Result {
        M::compile_wmma_fragment(f, fragment)
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &crate::shared::WmmaInstruction<Self>,
    ) -> std::fmt::Result {
        M::compile_wmma_instruction(f, instruction)
    }

    fn supported_wmma_combinations(
        arch: &AMDArchitecture,
    ) -> crate::shared::SupportedWmmaCombinations {
        M::supported_wmma_combinations(arch)
    }
}
