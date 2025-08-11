use crate::{MatrixLayout, TypeHash};

/// Hacky solution for getting comptime properties into the scope.
/// Allows querying certain target-specific properties at compile time, rather than at runtime.
/// Review on how to better solve this and delegate to the compiler if possible.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash, Default)]
pub struct TargetProperties {
    pub mma: MmaProperties,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
pub struct MmaProperties {
    /// Size of registers in bits, used to calculate line size
    pub register_size_bits: u32,
    /// Constant size of planes, for calculating lane indices in a matrix
    pub const_plane_size: u32,
    /// Layout of registers in Matrix A
    pub register_layout_a: MatrixLayout,
    /// Layout of registers in Matrix B
    pub register_layout_b: MatrixLayout,
    /// Layout of registers in Matrix C/D
    pub register_layout_acc: MatrixLayout,
}

impl Default for MmaProperties {
    fn default() -> Self {
        Self {
            register_size_bits: 32,
            const_plane_size: 32,
            register_layout_a: MatrixLayout::RowMajor,
            register_layout_b: MatrixLayout::ColMajor,
            register_layout_acc: MatrixLayout::RowMajor,
        }
    }
}
