use std::fmt::Display;

use type_hash::TypeHash;

use crate::{OperationReflect, Variable};

/// All metadata that can be accessed in a shader.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = MetadataOpCode)]
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Metadata::Rank { var } => write!(f, "rank({})", var),
            Metadata::Stride { dim, var } => write!(f, "{}.strides[{}]", var, dim),
            Metadata::Shape { dim, var } => write!(f, "{}.shape[{}]", var, dim),
            Metadata::Length { var } => write!(f, "{}.len()", var),
            Metadata::BufferLength { var } => write!(f, "buffer_len({})", var),
        }
    }
}
