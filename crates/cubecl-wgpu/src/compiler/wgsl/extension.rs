use hi_mul::*;

use super::base::Item;
use std::fmt::Display;

/// Not all functions are native to WGSL, so this struct allows to support more functions.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Extension {
    PowfScalar(Item),
    PowfPrimitive(Item),
    Powf(Item),
    Erf(Item),
    HiMul(Item),
    HiMul64(Item),
    HiMulSim(Item),
    #[cfg(target_os = "macos")]
    SafeTanh(Item),
}

impl Display for Extension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Extension::PowfScalar(elem) => format_powf_scalar(f, elem),
            Extension::PowfPrimitive(elem) => format_powf_primitive(f, elem),
            Extension::Powf(elem) => format_powf(f, elem),
            Extension::Erf(elem) => format_erf(f, elem),
            Extension::HiMul(elem) => format_hi_mul(f, elem),
            Extension::HiMul64(elem) => format_hi_mul_64(f, elem),
            Extension::HiMulSim(elem) => format_hi_mul_sim(f, elem),
            #[cfg(target_os = "macos")]
            Extension::SafeTanh(elem) => format_safe_tanh(f, elem),
        }
    }
}

mod hi_mul {
    use cubecl_core::ir::{ConstantScalarValue, IntKind, UIntKind};

    use crate::compiler::wgsl::Elem;

    use super::*;

    pub(crate) fn format_hi_mul_64(
        f: &mut std::fmt::Formatter<'_>,
        item: &Item,
    ) -> Result<(), std::fmt::Error> {
        let elem = item.elem();
        let elem64 = match elem {
            Elem::I32 => Elem::I64,
            Elem::U32 => Elem::U64,
            _ => unimplemented!("Hi mul only supports 32 bit ints"),
        };
        write!(
            f,
            "
    fn hi_mul_primitive(lhs: {elem}, rhs: {elem}) -> {elem} {{
        {elem}(({elem64}(lhs) * {elem64}(rhs)) >> 32)
    }}
    "
        )?;
        Ok(())
    }

    pub(crate) fn format_hi_mul_sim(
        f: &mut std::fmt::Formatter<'_>,
        item: &Item,
    ) -> Result<(), std::fmt::Error> {
        let elem = item.elem();
        let low_mask = match elem {
            Elem::U32 => ConstantScalarValue::UInt(0xffff, UIntKind::U32),
            Elem::I32 => ConstantScalarValue::Int(0xffff, IntKind::I32),
            _ => unimplemented!("Hi mul only supports 32 bit ints"),
        };
        write!(
            f,
            "
fn hi_mul_primitive(lhs: {elem}, rhs: {elem}) -> {elem} {{
    let lhs_low = lhs & {low_mask};
    let lhs_hi = (lhs >> 16) & {low_mask};
    let rhs_low = rhs & {low_mask};
    let rhs_hi = (rhs >> 16) & {low_mask};

    let low_low = lhs_low * rhs_low;
    let high_low = lhs_hi * rhs_low;
    let low_high = lhs_low * rhs_hi;
    let high_high = lhs_hi * rhs_hi;

    let mid = ((low_low >> 16) & {low_mask}) + (high_low & {low_mask}) + (low_high & {low_mask});
    return high_high + ((high_low >> 16) & {low_mask}) + ((low_high >> 16) & {low_mask}) + ((mid >> 16) & {low_mask});
}}
    "
        )?;
        Ok(())
    }

    pub(crate) fn format_hi_mul(
        f: &mut core::fmt::Formatter<'_>,
        item: &Item,
    ) -> core::fmt::Result {
        match item {
            Item::Vec4(_) => write!(
                f,
                "
fn hi_mul(lhs: {item}, rhs: {item}) -> {item} {{
    return vec4(
        hi_mul_primitive(lhs[0], rhs[0]),
        hi_mul_primitive(lhs[1], rhs[1]),
        hi_mul_primitive(lhs[2], rhs[2]),
        hi_mul_primitive(lhs[3], rhs[3]),
    );
}}
    "
            ),
            Item::Vec3(_) => write!(
                f,
                "
fn hi_mul(lhs: {item}, rhs: {item}) -> {item} {{
    return vec3(
        hi_mul_primitive(lhs[0], rhs[0]),
        hi_mul_primitive(lhs[1], rhs[1]),
        hi_mul_primitive(lhs[2], rhs[2]),
    );
}}
    "
            ),
            Item::Vec2(_) => write!(
                f,
                "
fn hi_mul(lhs: {item}, rhs: {item}) -> {item} {{
    return vec2(
        hi_mul_primitive(lhs[0], rhs[0]),
        hi_mul_primitive(lhs[1], rhs[1]),
    );
}}
    "
            ),
            Item::Scalar(elem) => write!(
                f,
                "
fn hi_mul(lhs: {elem}, rhs: {elem}) -> {elem} {{
    return hi_mul_primitive(lhs, rhs);
}}
    "
            ),
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

fn format_erf(f: &mut core::fmt::Formatter<'_>, ty: &Item) -> core::fmt::Result {
    let elem = ty.elem();
    write!(f,
        "
/// An approximation of the error function: https://en.wikipedia.org/wiki/Error_function#Numerical_approximations
///
/// > (maximum error: 1.5×10−7)
/// > All of these approximations are valid for x ≥ 0. To use these approximations for negative x, use the fact that erf x is an odd function, so erf x = −erf(−x).
fn erf_positive_scalar(x: {elem}) -> {elem} {{
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;

    let t = 1.0 / (1.0 + p * abs(x));
    let tmp = ((((a5 * t + a4) * t) + a3) * t + a2) * t + a1;

    return 1.0 - (tmp * t * exp(-x * x));
}}

fn erf_scalar(x: {elem}) -> {elem} {{
    if (x < 0.0) {{
        return -1.0 * erf_positive_scalar(-1.0 * x);
    }}

    return erf_positive_scalar(x);
}}
"
    )?;

    match ty {
        Item::Vec4(_) => write!(
            f,
            "
fn erf(x: {ty}) -> {ty} {{
    return vec4(
       erf_scalar(x[0]),
       erf_scalar(x[1]),
       erf_scalar(x[2]),
       erf_scalar(x[3]),
    );
}}
                "
        ),
        Item::Vec3(_) => write!(
            f,
            "
fn erf(x: {ty}) -> {ty} {{
    return vec3(
       erf_scalar(x[0]),
       erf_scalar(x[1]),
       erf_scalar(x[2]),
    );
}}
                "
        ),
        Item::Vec2(_) => write!(
            f,
            "
fn erf(x: {ty}) -> {ty} {{
    return vec2(
       erf_scalar(x[0]),
       erf_scalar(x[1]),
    );
}}
                "
        ),
        Item::Scalar(_) => write!(
            f,
            "
fn erf(x: {ty}) -> {ty} {{
   return erf_scalar(x);
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
