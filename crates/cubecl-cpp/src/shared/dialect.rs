use std::hash::Hash;
use std::{collections::HashSet, fmt::Debug};

use super::{
    Architecture, Binding, Elem, Flags, Fragment, FragmentIdent, FragmentLayout, Instruction, Item,
    SupportedWmmaCombinations, Variable, WmmaInstruction,
};

// Base dialect

pub trait Dialect:
    DialectIncludes<Self>
    + DialectTypes<Self>
    + DialectBindings<Self>
    + DialectCubeBuiltins
    + DialectInstructions<Self>
    + DialectWarp
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
    fn compile_instruction_sync_threads(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    fn compile_instruction_thread_fence(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

    fn compile_instruction_printf(
        f: &mut std::fmt::Formatter<'_>,
        format_string: &String,
        args: &Vec<Variable<D>>,
    ) -> std::fmt::Result;
}

// Warp

pub trait DialectWarp {
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
    fn compile_warp_ballot(f: &mut std::fmt::Formatter<'_>, var: &str) -> std::fmt::Result;
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
