use std::fmt::Display;

use crate::{
    shared::{
        self, AtomicKind, Binding, Component, DialectBindings, DialectCubeBuiltins, DialectIncludes, DialectInstructions, DialectTypes, DialectWarp, DialectWmmaCompiler, Elem, Flags, FmtLeft, Fragment, FragmentIdent, FragmentLayout, Instruction, Item, SupportedWmmaCombinations, Variable, WmmaInstruction
    }, Dialect
};

use super::{
    AddressSpace, Extension, arch::MetalArchitecture, format_erf, format_global_binding_arg,
    format_metal_builtin_binding_arg,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MslDialect {}

// Base dialect

impl Dialect for MslDialect {}

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
                Extension::NoExtension => {}
            }
        }
        Ok(())
    }

    fn register_extension(extensions: &mut Vec<Self::Extension>, instruction: &Instruction<Self>) {
        let mut register_extension = |extension: Self::Extension| {
            if !extensions.contains(&extension) {
                extensions.push(extension);
            }
        };
        #[allow(clippy::single_match)]
        match instruction {
            shared::Instruction::<Self>::Erf(instruction) => {
                register_extension(Extension::Erf(instruction.input, instruction.out));
            }
            _ => {}
        }
    }
}

// Types

impl DialectTypes<Self> for MslDialect {
    fn compile_type_definitions(
        f: &mut std::fmt::Formatter<'_>,
        items: &std::collections::HashSet<crate::shared::Item<Self>>,
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
    ) -> std::fmt::Result {
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

    fn compile_shared_memory_qualifier(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "threadgroup")
    }
}

// Kernel argument bindings

impl DialectBindings<Self> for MslDialect {
    fn compile_kernel_signature(
        f: &mut std::fmt::Formatter<'_>,
        kernel_name: &str,
        inputs: &[Binding<Self>],
        outputs: &[Binding<Self>],
        named: &[(String, Binding<Self>)],
        flags: &Flags,
    ) -> std::fmt::Result {
        write!(
            (f),
            "
[[kernel]]
void {}(",
            kernel_name
        )?;
        // Global bindings args
        let mut buffer_idx = 0;
        for (i, b) in inputs.iter().enumerate() {
            format_global_binding_arg("in", b, Some(&i.to_string()), buffer_idx, f)?;
            buffer_idx += 1;
        }
        for (i, b) in outputs.iter().enumerate() {
            format_global_binding_arg("out", b, Some(&i.to_string()), buffer_idx, f)?;
            buffer_idx += 1;
        }
        for (name, b) in named.iter() {
            format_global_binding_arg(name, b, None, buffer_idx, f)?;
            buffer_idx += 1;
        }
        // Global metal builtins args
        let builtins = vec![
            (flags.var_absolute_pos, Variable::<Self>::AbsolutePos),
            (flags.var_cube_dim, Variable::CubeDim),
            (flags.var_cube_count, Variable::CubeCount),
            (flags.var_unit_pos_global, Variable::UnitPosGlobal),
            (flags.var_unit_pos, Variable::UnitPos),
            (flags.var_cube_pos, Variable::CubePos),
            (flags.var_unit_pos_plane, Variable::UnitPosPlane),
            (flags.var_plane_dim, Variable::PlaneDim),
        ];
        let comma = !inputs.is_empty() || !outputs.is_empty() || !named.is_empty();
        builtins
            .iter()
            .filter(|(cond, _)| *cond)
            .try_for_each(|(_, var)| format_metal_builtin_binding_arg(f, var, comma))?;
        f.write_str("\n)")
    }
}

// Cube builtins dialect

impl DialectCubeBuiltins for MslDialect {
    fn compile_absolute_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_pos_in_grid")
    }

    fn compile_absolute_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_index_in_grid")
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
        f.write_str("threadgroups_per_grid")
    }

    fn compile_cube_count_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("total_threadgroups_in_grid")
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
        f.write_str("threads_per_threadgroup")
    }

    fn compile_cube_dim_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("total_thread_in_threadgroup")
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
        f.write_str("threadgroup_pos_in_grid")
    }

    fn compile_cube_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_index_in_grid")
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

    fn compile_unit_pos(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_pos_in_threadgroup")
    }

    fn compile_unit_pos_global(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("thread_index_in_threadgroup")
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
        format_string: &String,
        args: &Vec<Variable<Self>>,
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

    // sync
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "threadgroup_barrier(mem_flags::mem_threadgroup);")
    }

    fn compile_instruction_thread_fence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "threadgroup_thread_fence(mem_flags::mem_device);")
    }

    // unary
    fn compile_instruction_leading_zeros_scalar<T: Component<Self>>(f: &mut std::fmt::Formatter<'_>, input: T, output: Elem<Self>) -> std::fmt::Result {
        match input.elem() {
            Elem::I32 | Elem::U32 => write!(f, "static_cast<{output}>(clz({input}))"),
            Elem::I64 | Elem::U64 => panic!("leading_zeros instruction does not support 64-bit int"),
            elem => write!(
                f,
                "static_cast<{output}>(clz({})) - {}",
                shared::unary::zero_extend(input),
                (size_of::<u32>() - elem.size()) * 8
            ),
        }
    }

    fn compile_instruction_popcount_scalar<T: Component<Self>>(f: &mut std::fmt::Formatter<'_>, input: T, output: Elem<Self>) -> std::fmt::Result {
        match input.elem() {
            Elem::I32 | Elem::U32 => write!(f, "static_cast<{output}>(popcount({input}))"),
            Elem::I64 | Elem::U64 => panic!("popcount instruction does not support 64-bit int"),
            _ => write!(f, "static_cast<{output}>(popcount({}))", shared::unary::zero_extend(input)),
        }
    }

    fn compile_instruction_reverse_bits_scalar<T: Component<Self>>(f: &mut std::fmt::Formatter<'_>, input: T, output: Elem<Self>) -> std::fmt::Result {
        match output {
            Elem::I32 | Elem::U32 => write!(f, "reverse_bits({input})"),
            Elem::I64 | Elem::U64 => panic!("reverse_bits instruction does not support 64-bit int"),
            _ => write!(
                f,
                "{output}(reverse_bits({}) >> {})",
                shared::unary::zero_extend(input),
                (size_of::<u32>() - output.size()) * 8
            ),
        }
    }

}

// Warp

impl DialectWarp for MslDialect {
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
    fn compile_warp_all(f: &mut std::fmt::Formatter<'_>, var: &str) -> std::fmt::Result {
        write!(f, "simd_all({var})")
    }
    fn compile_warp_any(f: &mut std::fmt::Formatter<'_>, var: &str) -> std::fmt::Result {
        write!(f, "simd_any({var})")
    }
    fn compile_warp_ballot(f: &mut std::fmt::Formatter<'_>, out: &str) -> std::fmt::Result {
        write!(f, "simd_ballot({out})")
    }
}

// Coop Matrices dialect

impl DialectWmmaCompiler<Self> for MslDialect {
    type Architecture = MetalArchitecture;

    fn compile_wmma_includes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO
        println!("[compile_wmma_includes] NOT YET IMPLEMENTED");
        Ok(())
    }

    fn compile_wmma_type_definitions(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO
        println!("[compile_wmma_type_definitions] NOT YET IMPLEMENTED");
        Ok(())
    }

    fn compile_local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // TODO
        println!("[compile_local_variables] NOT YET IMPLEMENTED");
        Ok(())
    }

    fn compile_fragment_ident(
        _ident: &crate::shared::FragmentIdent<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // TODO
        println!("[compile_fragment_ident] NOT YET IMPLEMENTED");
        Ok(())
    }

    fn compile_fragment_layout(
        _layout: &crate::shared::FragmentLayout<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // TODO
        println!("[compile_fragment_layout] NOT YET IMPLEMENTED");
        Ok(())
    }

    fn compile_fragment(
        _fragment: &crate::shared::Fragment<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // TODO
        println!("[compile_fragment] NOT YET IMPLEMENTED");
        Ok(())
    }

    fn compile_instruction(
        _instruction: &crate::shared::WmmaInstruction<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        // TODO
        println!("[compile_instruction] NOT YET IMPLEMENTED");
        Ok(())
    }

    fn supported_wmma_combinations(
        _arch: &Self::Architecture,
    ) -> crate::shared::SupportedWmmaCombinations {
        vec![]
    }
}

// Coop Matrices dialect
