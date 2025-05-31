use crate::{
    Dialect,
    shared::{Elem, Item},
};

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension<D: Dialect> {
    Erf(Elem<D>, Elem<D>),
    Ffs(Elem<D>),
    MulHi(Elem<D>),
    SafeTanh(Item<D>),
    #[default]
    NoExtension,
}

pub fn format_erf<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input_elem: &Elem<D>,
    out_elem: &Elem<D>,
) -> core::fmt::Result {
    write!(
        f,
        "
// Abramowitz and Stegun approximation for erf(x)
inline {out_elem} erf({input_elem} x) {{
    const float a1 =  0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 =  1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 =  1.061405429f;
    const float p  =  0.3275911f;
    float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    x = fabs(x);
    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return sign * y;
}}
",
    )
}

pub fn format_ffs<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    input_elem: &Elem<D>,
) -> core::fmt::Result {
    match input_elem {
        Elem::I32 => write!(
            f,
            "
int __ffs(int x) {{
    return __ffs(static_cast<uint>(x));
}}
"
        ),
        Elem::U32 => write!(
            f,
            "
uint __ffs(uint x) {{
    return x == 0 ? 0 : 32 - clz(x & -x);
}}
"
        ),
        Elem::I64 => write!(
            f,
            "
int __ffsll(long x) {{
    return __ffsll(static_cast<ulong>(x));
}}
"
        ),
        Elem::U64 => write!(
            f,
            "
uint __ffsll(ulong x) {{
    return x == 0 ? 0 : 64 - clz(x & -x);
}}
"
        ),
        _ => Ok(()),
    }
}

pub fn format_mulhi<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    out_elem: &Elem<D>,
) -> core::fmt::Result {
    match out_elem {
        Elem::I32 => write!(
            f,
            "
int32_t __mulhi(int32_t a, int32_t b) {{
    int64_t product = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    return static_cast<int32_t>(product >> 32);
}}
"
        ),
        Elem::U32 => write!(
            f,
            "
uint32_t __umulhi(uint32_t a, uint32_t b) {{
    uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    return static_cast<uint32_t>(product >> 32);
}}
"
        ),
        Elem::I64 => write!(
            f,
            "
int64_t __mul64hi(int64_t a, int64_t b) {{
    // Determine the sign of the result
    bool negative = (a < 0) != (b < 0);
    // Compute absolute values
    uint64_t ua = static_cast<uint64_t>(a < 0 ? -a : a);
    uint64_t ub = static_cast<uint64_t>(b < 0 ? -b : b);
    // Perform unsigned high multiplication
    uint64_t high = __umul64hi(ua, ub);
    // Adjust sign if necessary
    return negative ? -static_cast<int64_t>(high) : static_cast<int64_t>(high);
}}
"
        ),
        Elem::U64 => write!(
            f,
            "
uint64_t __umul64hi(uint64_t a, uint64_t b) {{
    // Split the operands into high and low 32-bit parts
    uint64_t a_lo = static_cast<uint32_t>(a);
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = static_cast<uint32_t>(b);
    uint64_t b_hi = b >> 32;
    // Perform partial multiplications
    uint64_t p0 = a_lo * b_lo;
    uint64_t p1 = a_lo * b_hi;
    uint64_t p2 = a_hi * b_lo;
    uint64_t p3 = a_hi * b_hi;
    // Combine the results
    uint64_t mid = (p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF);
    uint64_t high = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return high;
}}
"
        ),
        _ => unimplemented!("HiMul only supports 32 and 64 bit ints"),
    }
}

pub fn format_safe_tanh<D: Dialect>(
    f: &mut core::fmt::Formatter<'_>,
    item: &Item<D>,
) -> core::fmt::Result {
    let elem = item.elem();
    write!(
        f,
        "
/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
inline {elem} safe_tanh_scalar({elem} x) {{
    if (x > 43.0) {{
        return 1.0;
    }} else {{
        return tanh(x);
    }}
}}
"
    )?;

    writeln!(f, "inline {item} safe_tanh({item} x) {{")?;
    if item.vectorization == 1 {
        writeln!(f, "    return safe_tanh_scalar(x);")?;
    } else {
        write!(f, "    return {item} {{ ")?;
        for i in 0..item.vectorization {
            let comma = if i != item.vectorization - 1 {
                ", "
            } else {
                ""
            };
            write!(f, "safe_tanh_scalar(x.i_{i}){comma}")?;
        }
        writeln!(f, " }};")?;
    }
    writeln!(f, "}}")
}
