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
    IsNanPrimitive(Elem),
    IsNan(Item, Item),
    IsInfPrimitive(Elem),
    IsInf(Item, Item),
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
                        item: *item,
                    },
                    VectorIdent {
                        name: "rhs",
                        item: *item,
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
                    item: *item,
                }],
                *item,
            ),
            Extension::IsNanPrimitive(elem) => format_is_nan_primitive(f, elem),
            Extension::IsNan(in_item, out_item) => construct_vector(
                f,
                IS_NAN,
                IS_NAN_PRIMITIVE,
                &[VectorIdent {
                    name: "x",
                    item: *in_item,
                }],
                *out_item,
            ),
            Extension::IsInfPrimitive(elem) => format_is_inf_primitive(f, elem),
            Extension::IsInf(in_item, out_item) => construct_vector(
                f,
                IS_INF,
                IS_INF_PRIMITIVE,
                &[VectorIdent {
                    name: "x",
                    item: *in_item,
                }],
                *out_item,
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

const IS_NAN_PRIMITIVE: &str = "is_nan_primitive";
const IS_NAN: &str = "is_nan";

const IS_INF_PRIMITIVE: &str = "is_inf_primitive";
const IS_INF: &str = "is_inf";

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
    let (rhs, base_name) = if should_use_scalar_powf(rhs) {
        let rhs = rhs.fmt_cast_to(Item::Scalar(lhs.elem()));
        (rhs, POWF_SCALAR)
    } else {
        let rhs = rhs.fmt_cast_to(lhs.item());
        (rhs, POWF)
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

pub fn call_is_nan(
    f: &mut core::fmt::Formatter,
    input: &Variable,
    out: &Variable,
) -> core::fmt::Result {
    let function_name = construct_vectorized_name(IS_NAN, input.item());
    let out = out.fmt_left();
    write!(f, "{out} = {function_name}({input});")
}

pub fn call_is_inf(
    f: &mut core::fmt::Formatter,
    input: &Variable,
    out: &Variable,
) -> core::fmt::Result {
    let function_name = construct_vectorized_name(IS_INF, input.item());
    let out = out.fmt_left();
    write!(f, "{out} = {function_name}({input});")
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
    let in_item = inputs[0].item;
    let vec_factor = output.vectorization_factor();
    let function_name = construct_vectorized_name(base_name, in_item);
    let primitive_name = construct_primitive_name(primitive_name, *in_item.elem());
    write!(f, "fn {function_name}(")?;
    for VectorIdent { name, item } in inputs {
        write!(f, "{name}: {item}, ")?;
    }
    writeln!(f, ") -> {output}{{")?;

    let indent = "    ";

    if let Item::Scalar(_) = output {
        write!(f, "{indent}return {primitive_name}(")?;
        for VectorIdent { name, item: _ } in inputs {
            write!(f, "{name}, ")?;
        }
        write!(f, ");\n}}\n")?;
        return Ok(());
    }

    writeln!(f, "{indent}return vec{vec_factor}(")?;

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
        writeln!(f, "),")?;
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

/// Returns (inf_bits, abs_mask, unsigned type) based on the floating point data type.
fn select_inf_bits_abs_mask_uint(in_elem: &Elem) -> (&'static str, &'static str, &'static str) {
    match in_elem {
        Elem::F16 => ("0x7c00", "0x7fff", "u16"),
        Elem::F32 => ("0x7f800000u", "0x7fffffffu", "u32"),
        Elem::F64 => ("0x7fffffffffffffff", "0x7ff0000000000000", "u64"),
        _ => unreachable!(),
    }
}

fn format_is_nan_primitive(
    f: &mut std::fmt::Formatter<'_>,
    in_elem: &Elem,
) -> Result<(), std::fmt::Error> {
    let function_name = construct_primitive_name(IS_NAN_PRIMITIVE, *in_elem);
    // Per NaN definition in IEEE 754:
    //   - sign = either 0 or 1.
    //   - biased exponent = all 1 bits.
    //   - fraction = anything except all 0 bits (since all 0 bits represents infinity).
    // https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
    let (inf_bits, abs_mask, uint) = select_inf_bits_abs_mask_uint(in_elem);
    write!(
        f,
        "
fn {function_name}(x: {in_elem}) -> bool {{
    let bits = bitcast<{uint}>(x);
    let abs_bits = bits & {abs_mask};
    return abs_bits > {inf_bits};
}}
"
    )?;
    Ok(())
}

fn format_is_inf_primitive(
    f: &mut std::fmt::Formatter<'_>,
    in_elem: &Elem,
) -> Result<(), std::fmt::Error> {
    let function_name = construct_primitive_name(IS_INF_PRIMITIVE, *in_elem);
    // Same trick as NaN detection following IEEE 754, but check for all 0 bits equality
    let (inf_bits, abs_mask, uint) = select_inf_bits_abs_mask_uint(in_elem);
    write!(
        f,
        "
fn {function_name}(x: {in_elem}) -> bool {{
    let bits = bitcast<{uint}>(x);
    let abs_bits = bits & {abs_mask};
    return abs_bits == {inf_bits};
}}
"
    )?;
    Ok(())
}
