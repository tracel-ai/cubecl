use super::base::Item;
use std::fmt::Display;

/// Not all functions are native to WGSL, so this struct allows to support more functions.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Extension {
    PowfScalar(Item),
    PowfPrimitive(Item),
    Powf(Item),
    #[cfg(target_os = "macos")]
    SafeTanh(Item),
}

impl Display for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Extension::PowfScalar(elem) => format_powf_scalar(f, elem),
            Extension::PowfPrimitive(elem) => format_powf_primitive(f, elem),
            Extension::Powf(elem) => format_powf(f, elem),
            #[cfg(target_os = "macos")]
            Extension::SafeTanh(elem) => format_safe_tanh(f, elem),
        }
    }
}

fn format_powf_scalar(f: &mut core::fmt::Formatter<'_>, item: &Item) -> core::fmt::Result {
    match item {
        Item::Vec4(elem) => write!(
            f,
            "
fn powf_scalar(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec4(
        powf_primitive(lhs[0], rhs),
        powf_primitive(lhs[1], rhs),
        powf_primitive(lhs[2], rhs),
        powf_primitive(lhs[3], rhs),
    );
}}
"
        ),
        Item::Vec3(elem) => write!(
            f,
            "
fn powf_scalar(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec3(
        powf_primitive(lhs[0], rhs),
        powf_primitive(lhs[1], rhs),
        powf_primitive(lhs[2], rhs),
    );
}}
"
        ),
        Item::Vec2(elem) => write!(
            f,
            "
fn powf_scalar(lhs: {item}, rhs: {elem}) -> {item} {{
    return vec2(
        powf_primitive(lhs[0], rhs),
        powf_primitive(lhs[1], rhs),
    );
}}
"
        ),
        Item::Scalar(elem) => write!(
            f,
            "
fn powf_scalar(lhs: {elem}, rhs: {elem}) -> {elem} {{
    return powf_primitive(lhs, rhs);
}}
"
        ),
    }
}

fn format_powf_primitive(
    f: &mut std::fmt::Formatter<'_>,
    item: &Item,
) -> Result<(), std::fmt::Error> {
    let elem = item.elem();
    write!(
        f,
        "
fn powf_primitive(lhs: {elem}, rhs: {elem}) -> {elem} {{
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

fn format_powf(f: &mut core::fmt::Formatter<'_>, item: &Item) -> core::fmt::Result {
    match item {
        Item::Vec4(_) => write!(
            f,
            "
fn powf(lhs: {item}, rhs: {item}) -> {item} {{
    return vec4(
        powf_primitive(lhs[0], rhs[0]),
        powf_primitive(lhs[1], rhs[1]),
        powf_primitive(lhs[2], rhs[2]),
        powf_primitive(lhs[3], rhs[3]),
    );
}}
"
        ),
        Item::Vec3(_) => write!(
            f,
            "
fn powf(lhs: {item}, rhs: {item}) -> {item} {{
    return vec3(
        powf_primitive(lhs[0], rhs[0]),
        powf_primitive(lhs[1], rhs[1]),
        powf_primitive(lhs[2], rhs[2]),
    );
}}
"
        ),
        Item::Vec2(_) => write!(
            f,
            "
fn powf(lhs: {item}, rhs: {item}) -> {item} {{
    return vec2(
        powf_primitive(lhs[0], rhs[0]),
        powf_primitive(lhs[1], rhs[1]),
    );
}}
"
        ),
        Item::Scalar(elem) => write!(
            f,
            "
fn powf(lhs: {elem}, rhs: {elem}) -> {elem} {{
    return powf_primitive(lhs, rhs);
}}
"
        ),
    }
}

#[cfg(target_os = "macos")]
fn format_safe_tanh(f: &mut core::fmt::Formatter<'_>, item: &Item) -> core::fmt::Result {
    let elem = item.elem();

    write!(
        f,
        "
/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
fn safe_tanh_scalar(x: {elem}) -> {elem} {{
    if x > 43.0 {{
        return 1.0;
    }} else {{
        return tanh(x);
    }}
}}
"
    )?;

    match item {
        Item::Vec4(_) => write!(
            f,
            "
fn safe_tanh(x: {item}) -> {item} {{
    return vec4(
        safe_tanh_scalar(x[0]),
        safe_tanh_scalar(x[1]),
        safe_tanh_scalar(x[2]),
        safe_tanh_scalar(x[3]),
    );
}}
"
        ),
        Item::Vec3(_) => write!(
            f,
            "
fn safe_tanh(x: {item}) -> {item} {{
    return vec3(
        safe_tanh_scalar(x[0]),
        safe_tanh_scalar(x[1]),
        safe_tanh_scalar(x[2]),
    );
}}
"
        ),
        Item::Vec2(_) => write!(
            f,
            "
fn safe_tanh(x: {item}) -> {item} {{
    return vec2(
        safe_tanh_scalar(x[0]),
        safe_tanh_scalar(x[1]),
    );
}}
"
        ),
        Item::Scalar(_) => write!(
            f,
            "
fn safe_tanh(x: {item}) -> {item} {{
    return safe_tanh_scalar(x);
}}
"
        ),
    }
}
