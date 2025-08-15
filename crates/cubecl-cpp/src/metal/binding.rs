use cubecl_core::compute::Visibility;

use crate::{
    Dialect,
    metal::AddressSpace,
    shared::{Binding, Component, MslComputeKernel, Variable},
};

pub fn bindings(repr: &MslComputeKernel) -> (Vec<Visibility>, Vec<Visibility>) {
    let mut bindings: Vec<Visibility> = vec![];
    // must be in the same order as the compilation order: inputs, outputs and named
    for b in repr.buffers.iter() {
        bindings.push(b.vis);
    }
    let mut meta: Vec<Visibility> = vec![];
    if repr.meta_static_len > 0 {
        meta.push(Visibility::Read);
    }
    for _ in repr.scalars.iter() {
        meta.push(Visibility::Read);
    }
    (bindings, meta)
}

pub fn format_global_binding_arg<D: Dialect>(
    name: &str,
    binding: &Binding<D>,
    suffix: Option<&str>,
    attr_idx: &mut usize,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let suffix = suffix.map_or("".into(), |s| format!("_{s}"));
    let (pointer, size) = match binding.size {
        Some(size) => ("".to_string(), format!("[{size}]")),
        None => (" *".to_string(), "".to_string()),
    };

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
    variable: &Variable<D>,
    comma: bool,
) -> core::fmt::Result {
    let ty = variable.item();
    let attribute = variable.attribute();
    let comma = if comma { "," } else { "" };
    write!(f, "{comma}\n    {ty} {variable} {attribute}",)?;
    Ok(())
}
