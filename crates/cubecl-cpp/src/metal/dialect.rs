use core::panic;
use std::fmt::Display;

use crate::{
    Dialect,
    shared::{
        self, AtomicKind, Binding, Component, CubeIndexFlags, DialectBindings, DialectCubeBuiltins,
        DialectIncludes, DialectInstructions, DialectTypes, DialectWmmaCompiler, Elem, Flags,
        FmtLeft, Fragment, FragmentIdent, FragmentLayout, Instruction, Item, SharedMemory,
        SupportedWmmaCombinations, Variable, WarpInstruction, WmmaInstruction, wmma_api_base,
    },
};
use cubecl_core::{
    compute::{Location, Visibility},
    ir::{self as gpu, Id},
};

use super::{
    AddressSpace, Extension,
    arch::MetalArchitecture,
    extension::{format_ffs, format_mulhi},
    format_erf, format_global_binding_arg, format_metal_builtin_binding_arg, format_safe_tanh,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MslDialect {}

// Base dialect

impl Dialect for MslDialect {
    type Architecture = MetalArchitecture;
}

// Includes

impl DialectIncludes<Self> for MslDialect {
    type Extension = Extension<Self>;

    fn compile_includes(f: &mut std::fmt::Formatter<'_>, _flags: &Flags) -> std::fmt::Result {
        write!(
            f,
            "
#include <metal_stdlib>
using namespace metal;
"
        )?;
        Ok(())
    }

    fn compile_extensions(
        f: &mut std::fmt::Formatter<'_>,
        extensions: &[Self::Extension],
    ) -> std::fmt::Result {
        for extension in extensions {
            match extension {
                Extension::Erf(input, output) => format_erf::<Self>(f, input, output)?,
                Extension::Ffs(elem) => format_ffs(f, elem)?,
                Extension::MulHi(elem) => format_mulhi(f, elem)?,
                Extension::SafeTanh(item) => format_safe_tanh::<Self>(f, item)?,
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
            shared::Instruction::<Self>::Erf(instruction) => {
                register_extension(Extension::Erf(
                    instruction.input.elem(),
                    instruction.out.elem(),
                ));
            }
            shared::Instruction::<Self>::FindFirstSet(instruction) => {
                let input_elem = instruction.input.elem();
                match input_elem {
                    Elem::U32 | Elem::U64 => {
                        register_extension(Extension::Ffs(instruction.input.elem()));
                    }
                    Elem::I32 => {
                        register_extension(Extension::Ffs(Elem::<Self>::U32));
                        register_extension(Extension::Ffs(instruction.input.elem()));
                    }
                    Elem::I64 => {
                        register_extension(Extension::Ffs(Elem::<Self>::U64));
                        register_extension(Extension::Ffs(instruction.input.elem()));
                    }
                    _ => {
                        register_extension(Extension::Ffs(Elem::<Self>::U32));
                    }
                }
            }
            shared::Instruction::<Self>::HiMul(instruction) => {
                register_extension(Extension::MulHi(instruction.out.elem()));
            }
            shared::Instruction::<Self>::Tanh(instruction) => {
                register_extension(Extension::SafeTanh(instruction.input.item()));
            }
            _ => {}
        }
    }

    fn register_warp_instruction_extension(
        _extensions: &mut Vec<Self::Extension>,
        _instruction: &WarpInstruction<Self>,
    ) {
    }
}

// Types

impl DialectTypes<Self> for MslDialect {
    fn item_can_be_optimized() -> bool {
        false
    }

    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &std::collections::HashSet<crate::shared::Item<Self>>,
        _scalars: &[(Elem<Self>, usize)],
        _flags: &Flags,
    ) -> std::fmt::Result {
        for item in items.iter() {
            let elem = item.elem;
            let size = item.vectorization;
            let alignment = elem.size() * size;
            if size > 1 {
                write!(
                    f,
                    "
struct alignas({alignment}) {item} {{"
                )?;

                for i in 0..size {
                    write!(
                        f,
                        "
    {elem} i_{i};"
                    )?;
                }

                f.write_str("\n};\n")?;
            }
        }
        Ok(())
    }

    fn compile_elem(
        f: &mut std::fmt::Formatter<'_>,
        elem: &shared::Elem<Self>,
        _words: bool,
    ) -> std::fmt::Result {
        // we always use the word form of types
        match elem {
            shared::Elem::F16 => f.write_str("half"),
            shared::Elem::F162 => panic!("type F162 not supported!"),
            shared::Elem::F32 => f.write_str("float"),
            shared::Elem::F64 => panic!("type double not supported!"),
            shared::Elem::BF16 => f.write_str("bfloat"),
            shared::Elem::BF162 => panic!("type BF162 not supported!"),
            shared::Elem::TF32 => f.write_str("float"),
            shared::Elem::I8 => f.write_str("char"),
            shared::Elem::I16 => f.write_str("short"),
            shared::Elem::I32 => f.write_str("int"),
            shared::Elem::I64 => f.write_str("long"),
            shared::Elem::U8 => f.write_str("uchar"),
            shared::Elem::U16 => f.write_str("ushort"),
            shared::Elem::U32 => f.write_str("uint"),
            shared::Elem::U64 => f.write_str("uint64_t"), // or unsigned long
            shared::Elem::Bool => f.write_str("bool"),
            shared::Elem::Atomic(inner) => inner.fmt(f),
            shared::Elem::_Dialect(_) => Ok(()),
        }
    }

    fn compile_item(f: &mut std::fmt::Formatter<'_>, item: &Item<Self>) -> std::fmt::Result {
        if 1 == item.vectorization {
            return write!(f, "{}", item.elem);
        }
        if item.native {
            write!(f, "{}{}", item.elem, item.vectorization)
        } else {
            write!(f, "{}_{}", item.elem, item.vectorization)
        }
    }

    fn compile_atomic_kind(
        f: &mut std::fmt::Formatter<'_>,
        kind: &AtomicKind<Self>,
    ) -> std::fmt::Result {
        match kind {
            AtomicKind::I32 => write!(f, "atomic_int"),
            AtomicKind::I64 => panic!("I64 atomic kind no supported."),
            AtomicKind::U32 => write!(f, "atomic_uint"),
            AtomicKind::U64 => write!(f, "atomic_ulong"),
            AtomicKind::F16 => panic!("F16 atomic kind no supported."),
            AtomicKind::BF16 => panic!("BF16 atomic kind no supported."),
            AtomicKind::F32 => write!(f, "atomic_float"), // needs metal 3
            AtomicKind::F64 => panic!("F64 atomic kind no supported."),
            AtomicKind::_Dialect(_) => Ok(()),
        }
    }

    fn address_space_for_variable(variable: &Variable<Self>) -> String {
        format!("{} ", AddressSpace::from(variable))
    }

    fn compile_local_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "thread")
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
            "threadgroup {alignment} {item} shared_memory_{index}[{size}];",
        )
    }
}

// Kernel argument bindings

impl DialectBindings<Self> for MslDialect {
    fn compile_kernel_signature(
        f: &mut std::fmt::Formatter<'_>,
        kernel_name: &str,
        tensor_maps: &[Id],
        buffers: &[Binding<Self>],
        scalars: &[(Elem<Self>, usize)],
        flags: &Flags,
    ) -> std::fmt::Result {
        write!(
            (f),
            "
[[kernel]]
void {kernel_name}("
        )?;
        // Global bindings args
        let mut buffer_idx = 0;
        debug_assert!(
            tensor_maps.is_empty(),
            "Tensor maps aren't supported for metal"
        );
        for (i, b) in buffers.iter().enumerate() {
            format_global_binding_arg("buffer", b, Some(&i.to_string()), &mut buffer_idx, f)?;
        }
        if flags.static_meta_length > 0 {
            let binding = Binding {
                id: 0,
                item: Item::scalar(Elem::<Self>::U32, true),
                location: Location::Storage,
                size: None,
                vis: Visibility::Read,
            };
            format_global_binding_arg("info", &binding, None, &mut buffer_idx, f)?;
        }
        for (elem, _) in scalars.iter() {
            let binding = Binding {
                id: 0,
                item: Item::scalar(*elem, true),
                location: Location::Storage,
                size: None,
                vis: Visibility::Read,
            };

            let name = format!("scalars_{elem}");
            format_global_binding_arg(&name, &binding, None, &mut buffer_idx, f)?;
        }

        // Global metal builtins args
        let builtins = vec![
            (
                flags.indexes.absolute_pos_tuple,
                Variable::<Self>::AbsolutePosBaseName,
            ),
            (
                flags.indexes.cube_dim_tuple,
                Variable::<Self>::CubeDimBaseName,
            ),
            (
                flags.indexes.cube_count_tuple,
                Variable::<Self>::CubeCountBaseName,
            ),
            (flags.indexes.unit_pos, Variable::<Self>::UnitPos),
            (
                flags.indexes.unit_pos_tuple,
                Variable::<Self>::UnitPosBaseName,
            ),
            (
                flags.indexes.cube_pos_tuple,
                Variable::<Self>::CubePosBaseName,
            ),
            (flags.indexes.unit_pos_plane, Variable::<Self>::UnitPosPlane),
            (flags.indexes.plane_dim, Variable::<Self>::PlaneDim),
            (flags.indexes.plane_index, Variable::<Self>::PlanePos),
        ];
        let comma = !buffers.is_empty() || flags.static_meta_length > 0 || !scalars.is_empty();
        builtins
            .iter()
            .filter(|(cond, _)| *cond)
            .try_for_each(|(_, var)| format_metal_builtin_binding_arg(f, var, comma))?;
        f.write_str("\n)")
    }
}

// Cube builtins dialect

impl DialectCubeBuiltins<Self> for MslDialect {
    /// Depending on the dialect available built-in variables the
    /// inclusion rules might change.
    /// For instance in metal we have a built-in for the Unit plane position
    /// so we don't rely on other builtins.
    fn builtin_rules(flags: &CubeIndexFlags) -> CubeIndexFlags {
        let absolute_pos = flags.absolute_pos;
        let cube_count = flags.cube_count;
        let cube_dim = flags.cube_dim;
        let cube_pos = flags.cube_pos;
        let plane_dim_checked = flags.plane_dim_checked;
        let plane_index = flags.plane_index;
        let unit_pos = flags.unit_pos;
        let absolute_pos_tuple = flags.absolute_pos_tuple || absolute_pos;
        let cube_count_tuple = flags.cube_count_tuple || cube_count || cube_pos || absolute_pos;
        let cube_dim_tuple = flags.cube_dim_tuple || cube_dim || absolute_pos || plane_dim_checked;
        let cube_pos_tuple = flags.cube_pos_tuple || cube_pos;
        let cluster_pos = flags.cluster_pos;
        let plane_dim = flags.plane_dim || plane_dim_checked || plane_index;
        let unit_pos_plane = flags.unit_pos_plane || plane_index;
        let unit_pos_tuple = flags.unit_pos_tuple || unit_pos;
        CubeIndexFlags {
            absolute_pos_tuple,
            absolute_pos,
            cube_count_tuple,
            cube_count,
            cube_dim_tuple,
            cube_dim,
            cube_pos_tuple,
            cube_pos,
            plane_dim,
            plane_dim_checked,
            plane_index,
            unit_pos_tuple,
            unit_pos,
            unit_pos_plane,
            cluster_pos,
        }
    }

    fn compile_absolute_pos_tuple_computation(
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // no need to compute it on metal as there is y a built-in for it
        Ok(())
    }

    fn compile_absolute_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_pos_in_grid")
    }

    fn compile_absolute_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_index_in_grid")
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
        f.write_str("threadgroups_per_grid")
    }

    fn compile_cube_count(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("total_threadgroups_in_grid")
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
        f.write_str("threads_per_threadgroup")
    }

    fn compile_cube_dim(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("total_thread_in_threadgroup")
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
        f.write_str("threadgroup_pos_in_grid")
    }

    fn compile_cube_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threadgroup_index_in_grid")
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

    fn compile_unit_pos_computation(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // no need to compute it on metal as there is y a built-in for it
        Ok(())
    }

    fn compile_unit_pos_base_name(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_pos_in_threadgroup")
    }

    fn compile_unit_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_index_in_threadgroup")
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
        f.write_str("simd_size")
    }

    fn compile_plane_dim_checked(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("threads_per_simdgroup_checked")
    }

    fn compile_plane_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("simd_group_id")
    }

    fn compile_unit_pos_plane(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("simd_lane_id")
    }
}

// Instructions

impl DialectInstructions<Self> for MslDialect {
    // atomics
    fn compile_atomic_add(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_fetch_add_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_and(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_fetch_and_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_cas(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<Self>,
        cmp: &Variable<Self>,
        val: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_compare_exchange_weak_explicit({input}, &{cmp}, {val}, memory_order_relaxed, memory_order_relaxed);"
        )
    }

    fn compile_atomic_load(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_load_explicit({input}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_max(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_fetch_max_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_min(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_fetch_min_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_or(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_fetch_or_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_store(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        writeln!(
            f,
            "atomic_store_explicit({out}, {input}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_sub(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_fetch_sub_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_swap(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_exchange_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    fn compile_atomic_xor(
        f: &mut std::fmt::Formatter<'_>,
        lhs: &Variable<Self>,
        rhs: &Variable<Self>,
        out: &Variable<Self>,
    ) -> std::fmt::Result {
        let out = out.fmt_left();
        writeln!(
            f,
            "{out} = atomic_fetch_xor_explicit({lhs}, {rhs}, memory_order_relaxed);"
        )
    }

    // debug
    fn compile_instruction_printf(
        f: &mut std::fmt::Formatter<'_>,
        format_string: &str,
        args: &[Variable<Self>],
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
        writeln!(f, "os_log_default.log(\"{format_string}\"{args});")
    }

    // logs
    fn compile_instruction_log1p_scalar<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
    ) -> std::fmt::Result {
        match input.elem() {
            Elem::F16 | Elem::F162 | Elem::BF16 | Elem::BF162 => {
                write!(f, "log(half(1.0f) + {input})")
            }
            _ => write!(f, "log(1.0f + {input})"),
        }
    }

    // sync
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "threadgroup_barrier(mem_flags::mem_threadgroup);")
    }

    fn compile_instruction_sync_warp(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "simdgroup_barrier(mem_flags::mem_none);")
    }

    fn compile_instruction_thread_fence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "threadgroup_thread_fence(mem_flags::mem_device);")
    }

    // trigo
    fn compile_instruction_tanh_scalar<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
    ) -> std::fmt::Result {
        write!(f, "safe_tanh_scalar({input})")
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
            _ => write!(f, "__ffs({}({input}))", Elem::<Self>::I32),
        }?;
        write!(f, ")")
    }

    fn compile_instruction_leading_zeros_scalar<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(clz({input}))")
    }

    fn compile_instruction_popcount_scalar<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match input.elem() {
            Elem::I32 | Elem::U32 | Elem::I64 | Elem::U64 => write!(f, "popcount({input})"),
            _ => write!(f, "popcount({})", shared::unary::zero_extend(input)),
        }?;
        write!(f, ")")
    }

    fn compile_instruction_reverse_bits_scalar<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: T,
        out_elem: Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(")?;
        match out_elem {
            Elem::I32 | Elem::U32 | Elem::I64 | Elem::U64 => write!(f, "reverse_bits({input})"),
            _ => write!(
                f,
                "reverse_bits({}) >> {}",
                shared::unary::zero_extend(input),
                (size_of::<u32>() - out_elem.size()) * 8
            ),
        }?;
        write!(f, ")")
    }

    // others
    fn compile_instruction_max_function_name(
        f: &mut std::fmt::Formatter<'_>,
        _item: Item<Self>,
    ) -> std::fmt::Result {
        write!(f, "max")
    }

    fn compile_instruction_min_function_name(
        f: &mut std::fmt::Formatter<'_>,
        _item: Item<Self>,
    ) -> std::fmt::Result {
        write!(f, "min")
    }

    fn compile_instruction_powf(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pow")
    }

    fn compile_instruction_half_function_name_prefix() -> &'static str {
        ""
    }

    fn compile_instruction_half2_function_name_prefix() -> &'static str {
        ""
    }

    // Warp
    fn compile_warp_shuffle(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        source: &str,
    ) -> std::fmt::Result {
        write!(f, "simd_shuffle({var}, {source})")
    }

    fn compile_warp_shuffle_xor(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        _elem: &Elem<Self>,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "simd_shuffle_xor({var}, {offset})")
    }

    fn compile_warp_shuffle_up(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "simd_shuffle_up({var}, {offset})")
    }

    fn compile_warp_shuffle_down(
        f: &mut std::fmt::Formatter<'_>,
        var: &str,
        offset: &str,
    ) -> std::fmt::Result {
        write!(f, "simd_shuffle_down({var}, {offset})")
    }

    fn compile_warp_all<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result {
        write!(f, "simd_all({input})")
    }

    fn compile_warp_any<T: Component<Self>>(
        f: &mut std::fmt::Formatter<'_>,
        input: &T,
    ) -> std::fmt::Result {
        write!(f, "simd_any({input})")
    }

    fn compile_warp_ballot(
        f: &mut std::fmt::Formatter<'_>,
        input: &Variable<Self>,
        out_elem: &Elem<Self>,
    ) -> std::fmt::Result {
        write!(f, "{out_elem}(uint64_t(simd_ballot({input})))")
    }
}

// Coop Matrices dialect

impl DialectWmmaCompiler<Self> for MslDialect {
    fn compile_wmma_includes(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "#include <metal_simdgroup_matrix>")
    }

    fn compile_wmma_type_definitions(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // not used
        Ok(())
    }

    fn compile_wmma_local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // not used
        Ok(())
    }

    fn compile_wmma_fragment_declaration(
        f: &mut std::fmt::Formatter<'_>,
        var: &crate::shared::Variable<MslDialect>,
    ) -> std::fmt::Result {
        wmma_api_base::compile_fragment_declaration(f, var)
    }

    fn compile_wwma_fragment_ident(
        _f: &mut std::fmt::Formatter<'_>,
        _ident: &FragmentIdent<Self>,
    ) -> std::fmt::Result {
        // not used
        Ok(())
    }

    fn compile_wmma_fragment_layout(
        _f: &mut std::fmt::Formatter<'_>,
        _layout: &FragmentLayout<Self>,
    ) -> std::fmt::Result {
        // not used
        Ok(())
    }

    fn compile_wmma_fragment(
        f: &mut std::fmt::Formatter<'_>,
        fragment: &Fragment<Self>,
    ) -> std::fmt::Result {
        let ty = fragment.elem;
        // currently as of Metal 3.2 only fragments of 8x8x8 are supported
        let m = fragment.m;
        let n = fragment.n;
        let k = fragment.k;
        if m != 8 || n != 8 || k != 8 {
            panic!("{m}x{n}x{k} fragments not supported. Only 8x8x8 fragments are supported.");
        }
        write!(f, "simdgroup_{ty}8x8")
    }

    fn compile_wmma_instruction(
        f: &mut std::fmt::Formatter<'_>,
        instruction: &WmmaInstruction<Self>,
    ) -> std::fmt::Result {
        match instruction {
            WmmaInstruction::Fill { frag, value } => {
                match frag {
                    Variable::WmmaFragment { .. } => {
                        let ty = frag.elem();
                        // Only 8x8x8 fragemts are supported. Check is done at fragment compilation time.
                        writeln!(
                            f,
                            "{frag} = make_filled_simdgroup_matrix<{ty}, 8, 8>({value});"
                        )
                    }
                    _ => panic!("should be a fragment"),
                }
            }
            WmmaInstruction::Load {
                frag,
                value,
                stride,
                offset,
                layout: _layout,
            } => {
                let transpose = match frag {
                    Variable::WmmaFragment { frag: inner, .. } => match inner.layout {
                        Some(FragmentLayout::RowMajor) => false,
                        Some(FragmentLayout::ColMajor) => true,
                        _ => false,
                    },
                    _ => panic!("should be a fragment"),
                };
                let item = value.item();
                if item.vectorization > 1 {
                    let elem = item.elem;
                    writeln!(
                        f,
                        "simdgroup_load({frag}, reinterpret_cast<threadgroup {elem} *>({value} + {offset}), {stride}, 0, {transpose});"
                    )
                } else {
                    writeln!(
                        f,
                        "simdgroup_load({frag}, {value} + {offset}, {stride}, 0, {transpose});"
                    )
                }
            }
            WmmaInstruction::Execute {
                frag_a: a,
                frag_b: b,
                frag_c: c,
                frag_d: d,
                ..
            } => {
                writeln!(f, "simdgroup_multiply_accumulate({d}, {a}, {b}, {c});")
            }
            WmmaInstruction::Store {
                output,
                frag,
                stride,
                offset,
                layout: _layout,
            } => {
                let item = output.item();
                let mut reinterpret_cast = item.vectorization > 1;
                let elem = match item.elem {
                    Elem::BF16 => {
                        reinterpret_cast = true;
                        Elem::F16
                    }
                    _ => item.elem,
                };
                if reinterpret_cast {
                    writeln!(
                        f,
                        "simdgroup_store({frag}, reinterpret_cast<threadgroup {elem} *>({output} + {offset}), {stride});"
                    )
                } else {
                    writeln!(f, "simdgroup_store({frag}, {output} + {offset}, {stride});")
                }?;
                writeln!(f, "simdgroup_barrier(mem_flags::mem_none);")
            }
            WmmaInstruction::Cast { input, output } => {
                writeln!(f, "simdgroup_barrier(mem_flags::mem_none);")?;
                let ty = match output {
                    Variable::WmmaFragment { frag, .. } => frag.elem,
                    _ => panic!("should be a fragment"),
                };
                match ty {
                    Elem::BF16 => {
                        let addr_space = Self::address_space_for_variable(output);
                        let elem = Elem::<Self>::F16;
                        // TODO: to test with benchmarks

                        writeln!(
                            f,
                            "for(int e=0; e<8; e++) {{
    {ty} elem = {ty}({input}.thread_elements()[e]);
    {output}.thread_elements()[e] = *reinterpret_cast<{addr_space}{elem} *>(&elem);
}}"
                        )
                    }
                    _ => {
                        writeln!(
                            f,
                            "for(int e=0; e<8; e++) {{
    {output}.thread_elements()[e] = {ty}({input}.thread_elements()[e]);
}}"
                        )
                    }
                }
            }
        }
    }

    fn supported_wmma_combinations(_arch: &MetalArchitecture) -> SupportedWmmaCombinations {
        vec![
            (
                gpu::Elem::Float(gpu::FloatKind::F16),
                gpu::Elem::Float(gpu::FloatKind::F16),
                gpu::Elem::Float(gpu::FloatKind::F16),
                vec![(8, 8, 8)],
            ),
            (
                gpu::Elem::Float(gpu::FloatKind::F16),
                gpu::Elem::Float(gpu::FloatKind::F16),
                gpu::Elem::Float(gpu::FloatKind::F32),
                vec![(8, 8, 8)],
            ),
            (
                gpu::Elem::Float(gpu::FloatKind::BF16),
                gpu::Elem::Float(gpu::FloatKind::BF16),
                gpu::Elem::Float(gpu::FloatKind::BF16),
                vec![(8, 8, 8)],
            ),
            (
                gpu::Elem::Float(gpu::FloatKind::F32),
                gpu::Elem::Float(gpu::FloatKind::F32),
                gpu::Elem::Float(gpu::FloatKind::F32),
                vec![(8, 8, 8)],
            ),
        ]
    }
}

// Coop Matrices dialect
