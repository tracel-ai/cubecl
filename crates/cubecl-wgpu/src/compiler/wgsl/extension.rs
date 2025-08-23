use super::base::{Elem, Item, Variable};
use std::fmt::Display;

/// Not all functions are native to WGSL, so this struct allows to support more functions.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Extension {
    PowfScalar(Item),
    PowfPrimitive(Elem),
    Powf(Item),
    #[cfg(target_os = "macos")]
    SafeTanh(Item),
    #[cfg(target_os = "macos")]
    SafeTanhPrimitive(Elem),
}

impl Display for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Extension::PowfPrimitive(elem) => format_powf_primitive(f, elem),
            Extension::PowfScalar(item) => construct_vector(
                f,
                POWF_SCALAR,
                POWF_PRIMITIVE,
                &[
                    VectorIdent {
                        name: "lhs",
                        item: *item,
                    },
                    VectorIdent {
                        name: "rhs",
                        item: Item::Scalar(*item.elem()),
                    },
                ],
                *item,
            ),
            Extension::Powf(item) => construct_vector(
                f,
                POWF,
                POWF_PRIMITIVE,
                &[
                    VectorIdent {
                        name: "lhs",
                        item: item.clone(),
                    },
                    VectorIdent {
                        name: "rhs",
                        item: item.clone(),
                    },
                ],
                *item,
            ),
            #[cfg(target_os = "macos")]
            Extension::SafeTanhPrimitive(elem) => format_safe_tanh_primitive(f, elem),
            #[cfg(target_os = "macos")]
            Extension::SafeTanh(item) => construct_vector(
                f,
                SAFE_TANH,
                SAFE_TANH_PRIMITIVE,
                &[VectorIdent {
                    name: "x",
                    item: item.clone(),
                }],
                *item,
            ),
        }
    }
}

const POWF_PRIMITIVE: &str = "powf_primitive";
const POWF: &str = "powf";
const POWF_SCALAR: &str = "powf_scalar";

#[cfg(target_os = "macos")]
const SAFE_TANH_PRIMITIVE: &str = "safe_tanh_primitive";
#[cfg(target_os = "macos")]
const SAFE_TANH: &str = "safe_tanh";

pub fn powf_extension(rhs: &Variable, out: &Variable) -> Extension {
    if should_use_scalar_powf(rhs) {
        Extension::PowfScalar(out.item())
    } else {
        Extension::Powf(out.item())
    }
}

pub fn call_powf(
    f: &mut core::fmt::Formatter,
    lhs: &Variable,
    rhs: &Variable,
    out: &Variable,
) -> core::fmt::Result {
    let base_name = if should_use_scalar_powf(rhs) {
        POWF_SCALAR
    } else {
        POWF
    };
    let function_name = construct_vectorized_name(base_name, out.item());

    let out = out.fmt_left();
    write!(f, "{out} = {function_name}({lhs}, {rhs});")
}

#[cfg(target_os = "macos")]
pub fn call_safe_tanh(
    f: &mut core::fmt::Formatter,
    input: &Variable,
    out: &Variable,
) -> core::fmt::Result {
    let function_name = construct_vectorized_name(SAFE_TANH, out.item());
    let out = out.fmt_left();
    write!(f, "{out} = {function_name}({input});")
}

fn should_use_scalar_powf(rhs: &Variable) -> bool {
    rhs.is_always_scalar() || rhs.item().vectorization_factor() == 1
}

fn construct_vectorized_name(base_name: &str, item: Item) -> String {
    let vec_factor = item.vectorization_factor();
    let elem = item.elem();
    format!("{base_name}_{vec_factor}_{elem}")
}

fn construct_primitive_name(base_name: &str, elem: Elem) -> String {
    format!("{base_name}_{elem}")
}

struct VectorIdent {
    name: &'static str,
    item: Item,
}

fn construct_vector(
    f: &mut core::fmt::Formatter<'_>,
    base_name: &str,
    primitive_name: &str,
    inputs: &[VectorIdent],
    output: Item,
) -> core::fmt::Result {
    let vec_factor = output.vectorization_factor();
    let function_name = construct_vectorized_name(base_name, output);
    let primitive_name = construct_primitive_name(primitive_name, *output.elem());
    write!(f, "fn {function_name}(")?;
    for VectorIdent { name, item } in inputs {
        write!(f, "{name}: {item}, ")?;
    }
    write!(f, ") -> {output}{{\n")?;

    let indent = "    ";

    match output {
        Item::Scalar(_) => {
            write!(f, "{indent}return {primitive_name}(")?;
            for VectorIdent { name, item: _ } in inputs {
                write!(f, "{name}, ")?;
            }
            write!(f, ");\n}}\n")?;
            return Ok(());
        }
        _ => {}
    }

    write!(f, "{indent}return vec{vec_factor}(\n")?;

    for i in 0..vec_factor {
        write!(f, "{indent}{indent}{primitive_name}(")?;
        for VectorIdent { name, item } in inputs {
            match item {
                Item::Scalar(_) => {
                    write!(f, "{name}, ")?;
                }
                _ => {
                    write!(f, "{name}[{i}], ")?;
                }
            }
        }
        write!(f, "),\n")?;
    }

    write!(f, "{indent});\n}}\n")?;

    Ok(())
}

fn format_powf_primitive(
    f: &mut std::fmt::Formatter<'_>,
    elem: &Elem,
) -> Result<(), std::fmt::Error> {
    let function_name = construct_primitive_name(POWF_PRIMITIVE, *elem);
    write!(
        f,
        "
fn {function_name}(lhs: {elem}, rhs: {elem}) -> {elem} {{
    let modulo = rhs % 2.0;
    if rhs == 0.0 {{
        return 1.0;
    }}
    if (modulo == 0.0) {{
        // Even number
        return pow(abs(lhs), rhs);
    }} else if (modulo == 1.0 && lhs < 0.0) {{
        // Odd number
        return -1.0 * pow(-1.0 * lhs, rhs);
    }} else {{
        // Float number
        return pow(lhs, rhs);
    }}
}}
"
    )?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn format_safe_tanh_primitive(
    f: &mut std::fmt::Formatter<'_>,
    elem: &Elem,
) -> Result<(), std::fmt::Error> {
    let function_name = construct_primitive_name(SAFE_TANH_PRIMITIVE, *elem);
    write!(
        f,
        "
/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
fn {function_name}(x: {elem}) -> {elem} {{
    if x > 43.0 {{
        return 1.0;
    }} else {{
        return tanh(x);
    }}
}}
"
    )?;
    Ok(())
}
