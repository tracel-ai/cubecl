use core::fmt::Display;

use cubecl_runtime::kernel::KernelArg;
use pliron::{context::Context, r#type::Typed};

use crate::{
    metal::{AddressSpace, BuiltInAttribute},
    shared::{
        CppValue,
        ty::{TypeExtCPP, TypedExtCPP},
    },
};

pub fn format_global_binding_arg(
    ctx: &Context,
    binding: &KernelArg,
    attr_idx: &mut usize,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let comma = if *attr_idx > 0 { "," } else { "" };
    let address_space = AddressSpace::from(binding.value.address_space_cpp(ctx));
    let ty = binding.value.get_type(ctx).to_cpp(ctx);
    let name = binding.value.name(ctx);
    let attribute = address_space.attribute();

    write!(f, "{comma}\n    {ty} {name}",)?;
    // attribute
    attribute.indexed_fmt(*attr_idx, f)?;
    *attr_idx += 1;
    Ok(())
}

pub fn format_metal_builtin_binding_arg(
    f: &mut core::fmt::Formatter<'_>,
    name: impl Display,
    attribute: &BuiltInAttribute,
    comma: bool,
) -> core::fmt::Result {
    let ty = attribute.cpp_ty();
    let comma = if comma { "," } else { "" };
    write!(f, "{comma}\n    {ty} {name} {attribute}",)?;
    Ok(())
}
