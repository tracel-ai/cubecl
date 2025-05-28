use crate::{Dialect, shared::Elem};

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension<D: Dialect> {
    F162BF16,
    Max(Elem<D>),
    Min(Elem<D>),
    #[default]
    NoExtension,
}

pub fn format_f162bf16(f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
        f,
        "
__device__ __bf16 half_to_bfloat16(__half h) {{
    float temp = float(h);
    return __bf16(temp);
}}
"
    )
}

pub fn format_max<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    elem: &Elem<D>,
) -> core::fmt::Result {
    match elem {
        crate::shared::Elem::BF16 => write!(
            f,
            "
__device__ __bf16 max_bfloat16(__bf16 a, __bf16 b) {{
    float fa = float(a);
    float fb = float(b);
    float max_val = fmaxf(fa, fb);
    return __bf16(max_val);
}}
"
        ),
        _ => Ok(()),
    }
}

pub fn format_min<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    elem: &Elem<D>,
) -> core::fmt::Result {
    match elem {
        crate::shared::Elem::BF16 => write!(
            f,
            "
__device__ __bf16 min_bfloat16(__bf16 a, __bf16 b) {{
    float fa = float(a);
    float fb = float(b);
    float min_val = fminf(fa, fb);
    return __bf16(min_val);
}}
"
        ),
        _ => Ok(()),
    }
}
