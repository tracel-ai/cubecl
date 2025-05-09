use core::fmt::Display;

use crate::TypeHash;

use crate::{OperationReflect, Variable};

/// All metadata that can be accessed in a shader.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = MetadataOpCode, pure)]
#[allow(missing_docs)]
pub enum Metadata {
    /// The rank of an array.
    Rank { var: Variable },
    /// The stride of an array at the given dimension.
    Stride { dim: Variable, var: Variable },
    /// The shape of an array at the given dimension.
    Shape { dim: Variable, var: Variable },
    /// The length of an array.
    Length { var: Variable },
    /// The length of an array's underlying buffer.
    BufferLength { var: Variable },
}

impl Display for Metadata {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Metadata::Rank { var } => write!(f, "rank({var})"),
            Metadata::Stride { dim, var } => write!(f, "{var}.strides[{dim}]"),
            Metadata::Shape { dim, var } => write!(f, "{var}.shape[{dim}]"),
            Metadata::Length { var } => write!(f, "{var}.len()"),
            Metadata::BufferLength { var } => write!(f, "buffer_len({var})"),
        }
    }
}
