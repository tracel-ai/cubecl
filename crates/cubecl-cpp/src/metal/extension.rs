use cubecl_core::ir::prelude::*;

use crate::shared::ty::TypeExtCPP;

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension {
    SafeTanh(TypeHandle),
    #[default]
    NoExtension,
}

pub fn format_safe_tanh(
    f: &mut core::fmt::Formatter<'_>,
    ctx: &Context,
    item: TypeHandle,
) -> core::fmt::Result {
    let elem = item.to_cpp(ctx);
    write!(
        f,
        "
/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
inline {elem} safe_tanh({elem} x) {{
    if (x > 43.0) {{
        return 1.0;
    }} else {{
        return tanh(x);
    }}
}}
"
    )
}
