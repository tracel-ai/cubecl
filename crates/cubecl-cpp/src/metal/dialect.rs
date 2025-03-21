use std::fmt::Display;

use crate::{
    Dialect,
    shared::{
        self, Binding, DialectBindings, DialectCubeBuiltins, DialectIncludes, DialectTypes,
        DialectWarp, DialectWmmaCompiler, Flags, Instruction, Item, Variable,
    },
};

use super::{
    Extension, arch::MetalArchitecture, format_erf, format_global_binding_arg,
    format_metal_builtin_binding_arg,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MslDialect {}

// Base dialect

impl Dialect for MslDialect {}

// Includes

impl DialectIncludes<Self> for MslDialect {
    type Extension = Extension;

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
                Extension::Erf => format_erf::<Self>(f)?,
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
            shared::Instruction::<Self>::Erf(_) => {
                register_extension(Extension::Erf);
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
        f.write_str("thread_pos_in_grid")
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
        f.write_str("thread_position_in_threadgroup")
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

// Warp

impl DialectWarp for MslDialect {
    fn compile_warp_shuffle(
        _f: &mut std::fmt::Formatter<'_>,
        _var: &str,
        _source: &str,
    ) -> std::fmt::Result {
        Ok(())
    }
    fn compile_warp_shuffle_xor(
        _f: &mut std::fmt::Formatter<'_>,
        _var: &str,
        _offset: &str,
    ) -> std::fmt::Result {
        Ok(())
    }
    fn compile_warp_shuffle_up(
        _f: &mut std::fmt::Formatter<'_>,
        _var: &str,
        _offset: &str,
    ) -> std::fmt::Result {
        Ok(())
    }
    fn compile_warp_shuffle_down(
        _f: &mut std::fmt::Formatter<'_>,
        _var: &str,
        _offset: &str,
    ) -> std::fmt::Result {
        Ok(())
    }
    fn compile_warp_all(_f: &mut std::fmt::Formatter<'_>, _var: &str) -> std::fmt::Result {
        Ok(())
    }
    fn compile_warp_any(_f: &mut std::fmt::Formatter<'_>, _var: &str) -> std::fmt::Result {
        Ok(())
    }
    fn compile_warp_ballot(_f: &mut std::fmt::Formatter<'_>, _out: &str) -> std::fmt::Result {
        Ok(())
    }
}

// Coop Matrices dialect

impl DialectWmmaCompiler<Self> for MslDialect {
    type Architecture = MetalArchitecture;

    fn compile_wmma_includes(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_wmma_type_definitions(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_local_variables(_f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }

    fn compile_fragment_ident(
        _ident: &crate::shared::FragmentIdent<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn compile_fragment_layout(
        _layout: &crate::shared::FragmentLayout<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn compile_fragment(
        _fragment: &crate::shared::Fragment<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn compile_instruction(
        _instruction: &crate::shared::WmmaInstruction<Self>,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Ok(())
    }

    fn supported_wmma_combinations(
        _arch: &Self::Architecture,
    ) -> crate::shared::SupportedWmmaCombinations {
        vec![]
    }
}

// Coop Matrices dialect
