use crate::{shared::{Component, Variable}, Dialect};

#[derive(Debug, Clone, Default, PartialEq)]
pub enum Extension<D: Dialect> {
    Erf(Variable<D>, Variable<D>),
    #[default]
    NoExtension,
}

pub fn format_erf<D: Dialect>(f: &mut core::fmt::Formatter<'_>, input: &Variable<D>, output: &Variable<D>) -> core::fmt::Result {
    let input_elem = input.elem();
    let output_elem = output.elem();
    write!(
        f,
        "
// Abramowitz and Stegun approximation for erf(x)
inline {output_elem} erf({input_elem} x) {{
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
