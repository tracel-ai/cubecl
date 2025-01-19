use serde::{Deserialize, Serialize};

bitflags::bitflags! {
    /// Unchecked optimizations for float operations. May cause precision differences, or undefined
    /// behaviour if the relevant conditions are not followed.
    #[derive(Default, Clone, Debug, Hash, PartialEq, Eq, Serialize, Deserialize)]
    pub struct FastMath: u32 {
        /// Disable unsafe optimizations
        const None = 0;
        /// Assume values are never `NaN`. If they are, the result is considered undefined behaviour.
        const NotNaN = 1;
        /// Assume values are never `Inf`/`-Inf`. If they are, the result is considered undefined
        /// behaviour.
        const NotInf = 1 << 1;
        /// Ignore sign on zero values.
        const UnsignedZero = 1 << 2;
        /// Allow swapping float division with a reciprocal, even if that swap would change precision.
        const AllowReciprocal = 1 << 3;
        /// Allow contracting float operations into fewer operations, even if the precision could
        /// change.
        const AllowContraction = 1 << 4;
        /// Allow reassociation for float operations, even if the precision could change.
        const AllowReassociation = 1 << 5;
        /// Allow all mathematical transformations for float operations, including contraction and
        /// reassociation, even if the precision could change.
        const AllowTransform = 1 << 6;
        /// Allow using slightly lower precision intrinsics (CUDA `--use_fast_math`)
        const ReducedPrecision = 1 << 7;
    }
}
