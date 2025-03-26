use std::hash::Hash;
use std::{collections::HashSet, fmt::Debug};

use crate::shared::FmtLeft;

use super::{
    Architecture, AtomicKind, Binding, Component, Elem, Flags, Fragment, FragmentIdent, FragmentLayout, Instruction, Item, SupportedWmmaCombinations, Variable, WmmaInstruction
};

// Base dialect

pub trait Dialect:
    DialectIncludes<Self>
    + DialectTypes<Self>
    + DialectBindings<Self>
    + DialectCubeBuiltins
    + DialectInstructions<Self>
    + DialectWmmaCompiler<Self>
    + Default
    + Clone
    + Copy
    + Debug
    + Send
    + Sync
    + Eq
    + Hash
    + 'static
{
}

// Includes

pub trait DialectIncludes<D: Dialect> {
    type Extension: Debug + Clone + Sync + Send;

    fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result;
    fn compile_extensions(
        f: &mut std::fmt::Formatter<'_>,
        extensions: &[Self::Extension],
    ) -> std::fmt::Result;
    fn register_extension(extensions: &mut Vec<Self::Extension>, instruction: &Instruction<D>);
}

// Types

pub trait DialectTypes<D: Dialect> {
    fn compile_elem(f: &mut std::fmt::Formatter<'_>, elem: &Elem<D>) -> std::fmt::Result;

    fn compile_atomic_kind(
        f: &mut std::fmt::Formatter<'_>,
        kind: &AtomicKind<D>,
    ) -> std::fmt::Result {
        match kind {
            AtomicKind::I32 => Elem::<D>::I32.fmt(f),
            AtomicKind::I64 => Elem::<D>::I64.fmt(f),
            AtomicKind::U32 => Elem::<D>::U32.fmt(f),
            AtomicKind::U64 => Elem::<D>::U64.fmt(f),
            AtomicKind::F16 => Elem::<D>::F16.fmt(f),
            AtomicKind::BF16 => Elem::<D>::BF16.fmt(f),
            AtomicKind::F32 => Elem::<D>::F32.fmt(f),
            AtomicKind::F64 => Elem::<D>::F64.fmt(f),
            AtomicKind::_Dialect(_) => Ok(()),
        }
    }

    fn compile_item(f: &mut std::fmt::Formatter<'_>, item: &Item<D>) -> std::fmt::Result;
    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &HashSet<Item<D>>,
        flags: &Flags,
    ) -> std::fmt::Result;
    fn compile_local_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_shared_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    /// Address space (for Metal dialect only).
    fn address_space_for_variable(_variable: &Variable<D>) -> String {
        "".to_string()
    }
}

// Kernel argument bindings

pub trait DialectBindings<D: Dialect> {
    fn compile_kernel_signature(
        f: &mut std::fmt::Formatter<'_>,
        kernel_name: &str,
        inputs: &[Binding<D>],
        outputs: &[Binding<D>],
        named: &[(String, Binding<D>)],
        flags: &Flags,
    ) -> std::fmt::Result;
}

// Cube builtins dialect

pub trait DialectCubeBuiltins {
    fn compile_absolute_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_absolute_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_absolute_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_absolute_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_absolute_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_count(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_count_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_count_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_count_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_count_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_dim(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_dim_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_dim_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_dim_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_dim_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_cube_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_unit_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_unit_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_unit_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_unit_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_unit_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
}

// Instructions

pub trait DialectInstructions<D: Dialect> {
    // atomics
    fn compile_atomic_add(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        match rhs.elem() {
            Elem::I64 => writeln!(
                f,
                "{out} = atomicAdd(reinterpret_cast<{uint}*>({lhs}), {uint}({rhs}));",
                uint = Elem::<D>::U64
            ),
            _ => writeln!(f, "{out} = atomicAdd({lhs}, {rhs});"),
        }
    }

    fn compile_atomic_and(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicAnd({lhs}, {rhs});")
    }

    fn compile_atomic_cas(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        cmp: &Variable<D>,
        val: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicCAS({input}, {cmp}, {val});")
    }

    fn compile_atomic_load(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicAdd({input}, 0);")
    }

    fn compile_atomic_max(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicMax({lhs}, {rhs});")
    }

    fn compile_atomic_min(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicMin({lhs}, {rhs});")
    }

    fn compile_atomic_or(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicOr({lhs}, {rhs});")
    }

    fn compile_atomic_store(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        writeln!(f, "atomicExch({out}, {input});")
    }

    fn compile_atomic_sub(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        match rhs.elem() {
            Elem::U32 | Elem::I32 => writeln!(f, "{out} = atomicSub({lhs}, {rhs});"),
            Elem::U64 => writeln!(f, "{out} = atomicAdd({lhs}, -{rhs});"),
            Elem::I64 => writeln!(
                f,
                "{out} = atomicAdd(reinterpret_cast<{uint}*>({lhs}), {uint}(-{rhs}));",
                uint = Elem::<D>::U64
            ),
            _ => writeln!(f, "{out} = atomicAdd({lhs}, -{rhs});"),
        }
    }

    fn compile_atomic_swap(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicExch({lhs}, {rhs});")
    }

    fn compile_atomic_xor(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<D>,
        rhs: &Variable<D>,
        out: &Variable<D>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(f, "{out} = atomicXor({lhs}, {rhs});")
    }

    // debug
    fn compile_instruction_printf(
        f: &mut std::fmt::Formatter<'_>,
        format_string: &String,
        args: &Vec<Variable<D>>,
    ) -> std::fmt::Result {
        let format_string = format_string
            .replace("\t", "\\t")
            .replace("\n", "\\n")
            .replace("\r", "\\r");
        let args = args.iter().map(|arg| format!("{arg}")).collect::<Vec<_>>();
        let args = match args.is_empty() {
            true => "".to_string(),
            false => format!(", {}", args.join(",")),
        };
        writeln!(f, "printf(\"{format_string}\"{args});")
    }

    // sync
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_instruction_thread_fence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    // unary
    fn compile_instruction_leading_zeros_scalar<T: Component<D>>(f: &mut std::fmt::Formatter<'_>, input: T, _output: Elem<D>) -> std::fmt::Result {
        match input.elem() {
            Elem::I32 | Elem::U32 => write!(f, "__clz({input})"),
            Elem::I64 | Elem::U64 => write!(f, "__clzll({input})"),
            elem => write!(
                f,
                "__clz({}) - {}",
                super::unary::zero_extend(input),
                (size_of::<u32>() - elem.size()) * 8
            ),
        }
    }

    fn compile_instruction_popcount_scalar<T: Component<D>>(f: &mut std::fmt::Formatter<'_>, input: T, _output: Elem<D>) -> std::fmt::Result {
        match input.elem() {
            Elem::I32 | Elem::U32 => write!(f, "__popc({input})"),
            Elem::I64 | Elem::U64 => write!(f, "__popcll({input})"),
            _ => write!(f, "__popc({})", super::unary::zero_extend(input)),
        }
    }

    fn compile_instruction_reverse_bits_scalar<T: Component<D>>(f: &mut std::fmt::Formatter<'_>, input: T, output: Elem<D>) -> std::fmt::Result {
        match output {
            Elem::I32 | Elem::U32 => write!(f, "__brev({input})"),
            Elem::I64 | Elem::U64 => write!(f, "__brevll({input})"),
            _ => write!(
                f,
                "{output}(__brev({}) >> {})",
                super::unary::zero_extend(input),
                (size_of::<u32>() - output.size()) * 8
            ),
        }
    }

    // warp
    fn compile_warp_shuffle(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        source: &str,
    ) -> std::fmt::Result;
    fn compile_warp_shuffle_xor(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result;
    fn compile_warp_shuffle_up(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result;
    fn compile_warp_shuffle_down(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result;
    fn compile_warp_all(f: &mut std::fmt::Formatter<'_>, var: &str) -> std::fmt::Result;
    fn compile_warp_any(f: &mut std::fmt::Formatter<'_>, var: &str) -> std::fmt::Result;
    fn compile_warp_ballot(f: &mut std::fmt::Formatter<'_>, input: &Variable<D>, _output: &Variable<D>) -> std::fmt::Result;
}

// Coop Matrices dialect

pub trait DialectWmmaCompiler<D: Dialect>:
    Default + Clone + Copy + Debug + Send + Sync + Eq + Hash + 'static
{
    type Architecture: Architecture;

    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_wmma_type_definitions(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_fragment_ident(
        ident: &FragmentIdent<D>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;
    fn compile_fragment_layout(
        layout: &FragmentLayout<D>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;
    fn compile_fragment(
        fragment: &Fragment<D>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;
    fn compile_instruction(
        instruction: &WmmaInstruction<D>,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;
    fn supported_wmma_combinations(arch: &Self::Architecture) -> SupportedWmmaCombinations;
}
