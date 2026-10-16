//! `ROCm` math library support.

use crate::shared::math_library::{FloatWidth, MathLibrary};

const NEVER_CORRECT: [&str; 9] = [
    "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "atan2", "pow",
];

const SINGLE_PRECISION_ONLY: [&str; 8] =
    ["exp", "exp2", "exp10", "log", "log2", "log10", "sin", "cos"];

pub struct Ocml;

impl MathLibrary for Ocml {
    /// The hardware's transcendentals take `f16` and `f32` only: LLVM cannot select them on a
    /// `bf16` or `f64` operand, so those go to the library.
    fn needs_redirect(&self, base: &str, width: FloatWidth) -> bool {
        let hardware = matches!(width, FloatWidth::F16 | FloatWidth::F32);
        NEVER_CORRECT.contains(&base) || (!hardware && SINGLE_PRECISION_ONLY.contains(&base))
    }

    /// ocml has no `bf16` functions, so a `bf16` call evaluates the `f32` one.
    fn symbol(&self, base: &str, width: FloatWidth) -> Option<(String, FloatWidth)> {
        let width = match width {
            FloatWidth::BF16 => FloatWidth::F32,
            width => width,
        };
        Some((format!("__ocml_{base}_{}", width.suffix()), width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_double_precision_needs_the_library_for_what_the_hardware_has() {
        assert!(!Ocml.needs_redirect("exp", FloatWidth::F32));
        assert!(Ocml.needs_redirect("exp", FloatWidth::F64));
        assert!(Ocml.needs_redirect("cos", FloatWidth::BF16));
        assert_eq!(
            Ocml.symbol("cos", FloatWidth::BF16),
            Some(("__ocml_cos_f32".to_string(), FloatWidth::F32))
        );
        assert!(Ocml.needs_redirect("atan2", FloatWidth::F32));
        assert!(Ocml.needs_redirect("atan2", FloatWidth::F64));
    }

    #[test]
    fn a_symbol_is_named_for_the_width_it_answers_at() {
        assert_eq!(
            Ocml.symbol("atan2", FloatWidth::F32),
            Some(("__ocml_atan2_f32".to_string(), FloatWidth::F32))
        );
    }
}
