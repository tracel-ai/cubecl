use cubecl_core::{prelude::Visibility, server::KernelArguments};

use crate::{
    Dialect,
    metal::AddressSpace,
    shared::{Component, KernelArg, MslComputeKernel, Variable},
};

pub fn bindings(repr: &MslComputeKernel, args: &KernelArguments) -> (Vec<Visibility>, usize) {
    let buffers = repr.buffers.iter().map(|it| it.vis);
    let uniform = args.info.dynamic_metadata_offset >= args.info.data.len();
    let info_vis = (!args.info.data.is_empty()).then_some(match uniform {
        true => Visibility::Uniform,
        false => Visibility::Read,
    });
    (buffers.chain(info_vis).collect(), 0)
}

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
    variable: &Variable<D>,
    comma: bool,
) -> core::fmt::Result {
    let ty = variable.item();
    let attribute = variable.attribute();
    let comma = if comma { "," } else { "" };
    write!(f, "{comma}\n    {ty} {variable} {attribute}",)?;
    Ok(())
}
