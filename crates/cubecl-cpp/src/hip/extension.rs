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
__device__ __hip_bfloat16 half_to_bfloat16(__half h) {{
    float temp = __half2float(h);
    return __float2bfloat16(temp);
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
__device__ __hip_bfloat16 max_bfloat16(__hip_bfloat16 a, __hip_bfloat16 b) {{
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    float max_val = fmaxf(fa, fb);
    return __float2bfloat16(max_val);
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
__device__ __hip_bfloat16 min_bfloat16(__hip_bfloat16 a, __hip_bfloat16 b) {{
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    float min_val = fminf(fa, fb);
    return __float2bfloat16(min_val);
}}
"
        ),
        _ => Ok(()),
    }
}
