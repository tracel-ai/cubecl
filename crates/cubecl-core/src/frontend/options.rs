use enumset::{EnumSet, EnumSetType};
use serde::{Deserialize, Serialize};

/// Unchecked optimizations for float operations. May cause precision differences, or undefined
/// behaviour if the relevant conditions are not followed.
#[derive(Default, Debug, Hash, Serialize, Deserialize, EnumSetType)]
pub enum FastMath {
    /// Disable unsafe optimizations
    #[default]
    None,
    /// Assume values are never `NaN`. If they are, the result is considered undefined behaviour.
    NotNaN,
    /// Assume values are never `Inf`/`-Inf`. If they are, the result is considered undefined
    /// behaviour.
    NotInf,
    /// Ignore sign on zero values.
    UnsignedZero,
    /// Allow swapping float division with a reciprocal, even if that swap would change precision.
    AllowReciprocal,
    /// Allow contracting float operations into fewer operations, even if the precision could
    /// change.
    AllowContraction,
    /// Allow reassociation for float operations, even if the precision could change.
    AllowReassociation,
    /// Allow all mathematical transformations for float operations, including contraction and
    /// reassociation, even if the precision could change.
    AllowTransform,
    /// Allow using lower precision intrinsics (CUDA `--use_fast_math`)
    /// Also impacts `NaN`, `Inf` and signed zero handling, as well as subnormals and rounding.
    ///
    /// Notable edge case:
    /// powf - Returns `NaN` for negative bases
    ReducedPrecision,
}

impl FastMath {
    pub fn all() -> EnumSet<FastMath> {
        EnumSet::all()
    }
}
