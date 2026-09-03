//! `ROCm`'s OCML, as [`MathLibrary`] describes it.
//!
//! The walk that finds the intrinsics and builds the wrappers is
//! [`shared::math_library`](crate::shared::math_library); what is here is the two things OCML
//! answers differently from `libdevice`: which intrinsics the AMDGPU backend cannot handle on
//! its own, and what OCML calls them.

use crate::shared::math_library::{FloatWidth, MathLibrary};

/// Intrinsics the backend cannot give a correct answer for at any width.
const NEVER_CORRECT: [&str; 9] = [
    "tan", "sinh", "cosh", "tanh", "asin", "acos", "atan", "atan2", "pow",
];

/// Intrinsics the hardware has in single precision but not in double, where the whole
/// transcendental unit is missing.
const SINGLE_PRECISION_ONLY: [&str; 8] =
    ["exp", "exp2", "exp10", "log", "log2", "log10", "sin", "cos"];

pub struct Ocml;

impl MathLibrary for Ocml {
    fn needs_redirect(&self, base: &str, width: FloatWidth) -> bool {
        NEVER_CORRECT.contains(&base)
            || (width == FloatWidth::F64 && SINGLE_PRECISION_ONLY.contains(&base))
    }

    /// OCML has an entry point at every width, named for it, so nothing is ever answered at a
    /// wider one.
    fn symbol(&self, base: &str, width: FloatWidth) -> Option<(String, FloatWidth)> {
        Some((format!("__ocml_{base}_{}", width.suffix()), width))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The single-precision transcendentals are the hardware's own, and only the double
    /// precision ones need the library. Redirecting the single-precision ones as well would
    /// give up the transcendental unit for a call.
    #[test]
    fn only_double_precision_needs_the_library_for_what_the_hardware_has() {
        assert!(!Ocml.needs_redirect("exp", FloatWidth::F32));
        assert!(Ocml.needs_redirect("exp", FloatWidth::F64));
        // Nothing is behind these at any width.
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
