use crate::{
    Dialect,
    metal::AddressSpace,
    shared::{Builtin, KernelArg},
};

pub fn format_global_binding_arg<D: Dialect>(
    name: &str,
    binding: &KernelArg<D>,
    suffix: Option<&str>,
    attr_idx: &mut usize,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let suffix = suffix.map_or("".into(), |s| format!("_{s}"));
    let (pointer, size) = (" *".to_string(), "".to_string());

    let comma = if *attr_idx > 0 { "," } else { "" };
    let address_space = AddressSpace::from(binding);
    let ty = binding.item;
    let attribute = address_space.attribute();

    write!(
        f,
        "{comma}\n    {address_space} {ty}{pointer} {name}{suffix}",
    )?;
    // attribute
    attribute.indexed_fmt(*attr_idx, f)?;
    write!(f, "{size}")?;
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
