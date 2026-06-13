use crate::{
    Dialect,
    metal::AddressSpace,
    shared::{Builtin, Component, KernelArg},
};

pub fn format_global_binding_arg<D: Dialect>(
    binding: &KernelArg<D>,
    attr_idx: &mut usize,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let comma = if *attr_idx > 0 { "," } else { "" };
    let address_space = AddressSpace::from(binding);
    let ty = binding.value.item();
    let name = binding.value;
    let attribute = address_space.attribute();

    write!(f, "{comma}\n    {ty} {name}",)?;
    // attribute
    attribute.indexed_fmt(*attr_idx, f)?;
    *attr_idx += 1;
    Ok(())
}

pub fn format_metal_builtin_binding_arg<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    builtin: &Builtin<D>,
    comma: bool,
) -> core::fmt::Result {
    let ty = builtin.item();
    let attribute = builtin.attribute();
    let comma = if comma { "," } else { "" };
    write!(f, "{comma}\n    {ty} {builtin} {attribute}",)?;
    Ok(())
}
