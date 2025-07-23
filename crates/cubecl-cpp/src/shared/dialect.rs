use std::hash::Hash;
use std::{collections::HashSet, fmt::Debug};

use cubecl_core::ir::Id;

use crate::shared::FmtLeft;

use super::{
    Architecture, AtomicKind, Binding, Body, Component, CubeIndexFlags, Elem, Flags, Fragment,
    FragmentIdent, FragmentLayout, Instruction, Item, SharedMemory, SupportedWmmaCombinations,
    Variable, WarpInstruction, WmmaInstruction,
};

// Base dialect

pub trait Dialect:
    DialectIncludes<Self>
    + DialectTypes<Self>
    + DialectBindings<Self>
    + DialectCubeBuiltins<Self>
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
    type Architecture: Architecture;
}

// Includes

pub trait DialectIncludes<D: Dialect> {
    type Extension: Debug + Clone + Sync + Send;

    fn compile_includes(f: &mut std::fmt::Formatter<'_>, flags: &Flags) -> std::fmt::Result;
    fn compile_extensions(
        f: &mut std::fmt::Formatter<'_>,
        extensions: &[Self::Extension],
    ) -> std::fmt::Result;
    fn register_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &Instruction<D>,
    );
    fn register_warp_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &WarpInstruction<D>,
    );
    #[allow(unused_variables)]
    fn register_wmma_instruction_extension(
        extensions: &mut Vec<Self::Extension>,
        instruction: &WmmaInstruction<D>,
    ) {
    }
}

// Types

pub trait DialectTypes<D: Dialect> {
    fn item_can_be_optimized() -> bool;
    fn compile_elem(
        f: &mut std::fmt::Formatter<'_>,
        elem: &Elem<D>,
        word: bool,
    ) -> std::fmt::Result;

    fn compile_atomic_kind(
        f: &mut std::fmt::Formatter<'_>,
        kind: &AtomicKind<D>,
    ) -> std::fmt::Result {
        match kind {
            AtomicKind::I32 => write!(f, "{}", Elem::<D>::I32),
            AtomicKind::I64 => write!(f, "{}", Elem::<D>::I64),
            AtomicKind::U32 => write!(f, "{}", Elem::<D>::U32),
            AtomicKind::U64 => write!(f, "{}", Elem::<D>::U64),
            AtomicKind::F16 => write!(f, "{}", Elem::<D>::F16),
            AtomicKind::BF16 => write!(f, "{}", Elem::<D>::BF16),
            AtomicKind::F32 => write!(f, "{}", Elem::<D>::F32),
            AtomicKind::F64 => write!(f, "{}", Elem::<D>::F64),
            AtomicKind::_Dialect(_) => Ok(()),
        }
    }

    fn compile_item(f: &mut std::fmt::Formatter<'_>, item: &Item<D>) -> std::fmt::Result;
    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &HashSet<Item<D>>,
        scalars: &[(Elem<D>, usize)],
        flags: &Flags,
    ) -> std::fmt::Result;
    fn compile_local_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_shared_memory_declaration(
        f: &mut std::fmt::Formatter<'_>,
        shared: &SharedMemory<D>,
    ) -> std::fmt::Result;
    fn compile_polyfills(_f: &mut std::fmt::Formatter<'_>, _flags: &Flags) -> std::fmt::Result {
        Ok(())
    }
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
        tensor_maps: &[Id],
        buffers: &[Binding<D>],
        scalars: &[(Elem<D>, usize)],
        flags: &Flags,
    ) -> std::fmt::Result;
    fn compile_bindings_body(
        _f: &mut std::fmt::Formatter<'_>,
        _body: &Body<D>,
    ) -> std::fmt::Result {
        Ok(())
    }
}

// Cube builtins dialect

pub trait DialectCubeBuiltins<D: Dialect> {
    /// Depending on the dialect available built-in variables the
    /// inclusion rules might change.
    /// For instance in metal we have a built-in for the Unit plane position
    /// but in other dialects there is none so we have to compute it using
    /// other built-ins.
    fn builtin_rules(flags: &CubeIndexFlags) -> CubeIndexFlags {
        let unit_pos_plane = flags.unit_pos_plane;
        let plane_dim_checked = flags.plane_dim_checked;
        let plane_dim = flags.plane_dim || plane_dim_checked || unit_pos_plane;
        let plane_index = flags.plane_index;
        let absolute_pos = flags.absolute_pos || unit_pos_plane;
        let absolute_pos_tuple = flags.absolute_pos_tuple || absolute_pos;
        let cube_dim = flags.cube_dim;
        let cube_dim_tuple = flags.cube_dim_tuple || cube_dim || absolute_pos || plane_dim_checked;
        let unit_pos = flags.unit_pos;
        let unit_pos_tuple = flags.unit_pos_tuple || unit_pos;
        let cube_count = flags.cube_count;
        let cube_count_tuple = flags.cube_count_tuple || absolute_pos;
        let cube_pos = flags.cube_pos;
        let cube_pos_tuple = flags.cube_pos_tuple || cube_pos;
        let cluster_group = flags.cluster_pos;

        CubeIndexFlags {
            absolute_pos,
            absolute_pos_tuple,
            cube_count,
            cube_count_tuple,
            cube_dim,
            cube_dim_tuple,
            cube_pos,
            cube_pos_tuple,
            plane_dim,
            plane_dim_checked,
            plane_index,
            unit_pos_tuple,
            unit_pos,
            unit_pos_plane,
            cluster_pos: cluster_group,
        }
    }

    fn compile_absolute_pos_tuple_computation(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variable = Variable::<D>::AbsolutePosBaseName;
        let ty = variable.item();
        let cube_pos_x = Variable::<D>::CubePosX;
        let cube_pos_y = Variable::<D>::CubePosY;
        let cube_pos_z = Variable::<D>::CubePosZ;
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        let cube_dim_z = Variable::<D>::CubeDimZ;
        let unit_pos_x = Variable::<D>::UnitPosX;
        let unit_pos_y = Variable::<D>::UnitPosY;
        let unit_pos_z = Variable::<D>::UnitPosZ;
        writeln!(
            f,
            "{ty} {variable} = make_{ty}(
    {cube_pos_x} * {cube_dim_x} + {unit_pos_x},
    {cube_pos_y} * {cube_dim_y} + {unit_pos_y},
    {cube_pos_z} * {cube_dim_z} + {unit_pos_z}
);"
        )
    }

    fn compile_absolute_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("absoluteIdx")
    }

    fn compile_absolute_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("idxGlobal")
    }

    fn compile_absolute_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_absolute_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_absolute_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_absolute_pos_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_cube_count_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("gridDim")
    }

    fn compile_cube_count(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("gridDimGlobal")
    }

    fn compile_cube_count_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_cube_count_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_cube_count_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_count_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_cube_dim_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockDim")
    }

    fn compile_cube_dim(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockDimGlobal")
    }

    fn compile_cube_dim_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_cube_dim_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_cube_dim_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_dim_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_cube_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockIdx")
    }

    fn compile_cube_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("blockIdxGlobal")
    }

    fn compile_cube_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_cube_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_cube_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_cube_pos_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_unit_pos_computation(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variable = Variable::<D>::UnitPos;
        let ty = variable.item();
        let cube_dim_x = Variable::<D>::CubeDimX;
        let cube_dim_y = Variable::<D>::CubeDimY;
        let unit_pos_x = Variable::<D>::UnitPosX;
        let unit_pos_y = Variable::<D>::UnitPosY;
        let unit_pos_z = Variable::<D>::UnitPosZ;
        writeln!(
            f,
            "{ty} {variable} = {unit_pos_x} + {unit_pos_y} * {cube_dim_x} + {unit_pos_z} * ({cube_dim_x} * {cube_dim_y});"
        )
    }

    fn compile_unit_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threadIdxGlobal")
    }

    fn compile_unit_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threadIdx")
    }

    fn compile_unit_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos_base_name(f)?;
        write!(f, ".x")
    }

    fn compile_unit_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos_base_name(f)?;
        write!(f, ".y")
    }

    fn compile_unit_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::compile_unit_pos_base_name(f)?;
        write!(f, ".z")
    }

    fn compile_plane_dim(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("warpSize")
    }

    fn compile_plane_dim_checked(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("warpSizeChecked")
    }

    fn compile_plane_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let unit_pos_x = Variable::<D>::UnitPosX;
        let plane_dim = Variable::<D>::PlaneDim;
        write!(f, "{unit_pos_x} / {plane_dim}")
    }

    fn compile_unit_pos_plane(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let absolute_pos = Variable::<D>::AbsolutePos;
        let plane_dim = Variable::<D>::PlaneDim;
        write!(f, "{absolute_pos} % {plane_dim}")
    }

    fn compile_cluster_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_x(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_y(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
    fn compile_cluster_pos_z(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0")
    }
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
        format_string: &str,
        args: &[Variable<D>],
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

    // logs
    fn compile_instruction_log1p_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
    ) -> std::fmt::Result {
        let elem = input.elem();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                write!(f, "{elem}(log1p(float({input})))")
            }
            _ => write!(f, "log1p({input})"),
        }
    }

    // sync
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_instruction_sync_warp(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_instruction_thread_fence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    // trigo
    fn compile_instruction_tanh_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
    ) -> std::fmt::Result {
        let elem = input.elem();
        match elem {
            Elem::F16 | Elem::F16x2 | Elem::BF16 | Elem::BF16x2 => {
                write!(f, "{elem}(tanh(float({input})))")
            }
            _ => write!(f, "tanh({input})"),
        }
    }

    // unary
    fn compile_instruction_find_first_set<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result;
    fn compile_instruction_leading_zeros_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result;

    fn compile_instruction_popcount_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match input.elem() {
            Elem::I32 => write!(f, "__popc({}({input}))", Elem::<D>::U32),
            Elem::U32 => write!(f, "__popc({input})"),
            Elem::I64 => write!(f, "__popcll({}({input}))", Elem::<D>::U64),
            Elem::U64 => write!(f, "__popcll({input})"),
            _ => write!(f, "__popc({})", super::unary::zero_extend(input)),
        }?;
        write!(f, ")")
    }

    fn compile_instruction_reverse_bits_scalar<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<D>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match out_elem {
            Elem::I32 => write!(f, "__brev({}({input}))", Elem::<D>::U32),
            Elem::U32 => write!(f, "__brev({input})"),
            Elem::I64 => write!(f, "__brevll({}({input}))", Elem::<D>::U64),
            Elem::U64 => write!(f, "__brevll({input})"),
            _ => write!(
                f,
                "__brev({}) >> {}",
                super::unary::zero_extend(input),
                (size_of::<u32>() - out_elem.size()) * 8
            ),
        }?;
        write!(f, ")")
    }

    // others
    fn compile_instruction_max_function_name(
        f: &mut std::fmt::Formatter<'_>,
        item: Item<D>,
    ) -> std::fmt::Result;

    fn compile_instruction_min_function_name(
        f: &mut std::fmt::Formatter<'_>,
        item: Item<D>,
    ) -> std::fmt::Result;

    fn compile_instruction_powf(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "powf")
    }

    fn compile_instruction_half_function_name_prefix() -> &'static str {
        "h"
    }

    fn compile_instruction_half2_function_name_prefix() -> &'static str {
        "h2"
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
        elem: &Elem<D>,
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
    fn compile_warp_all<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result;
    fn compile_warp_any<T: Component<D>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result;
    fn compile_warp_ballot(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<D>,
        out_elem: &Elem<D>,
    ) -> std::fmt::Result;
}

// Coop Matrices dialect

pub trait DialectWmmaCompiler<D: Dialect>:
    Default + Clone + Copy + Debug + Send + Sync + Eq + Hash + 'static
{
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_wmma_type_definitions(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_wmma_local_variables(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &Variable<D>,
    ) -> std::fmt::Result;
    fn compile_wwma_fragment_ident(
        f: &mut std::fmt::Formatter<'_>,
        ident: &FragmentIdent<D>,
    ) -> std::fmt::Result;
    fn compile_wmma_fragment_layout(
        f: &mut std::fmt::Formatter<'_>,
        layout: &FragmentLayout<D>,
    ) -> std::fmt::Result;
    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<D>,
    ) -> std::fmt::Result;
    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<D>,
    ) -> std::fmt::Result;
    fn supported_wmma_combinations(arch: &D::Architecture) -> SupportedWmmaCombinations;
}
