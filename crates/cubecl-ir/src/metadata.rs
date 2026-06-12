use core::fmt::Display;

use crate::TypeHash;

use crate::{OperationReflect, Value};

/// All metadata that can be accessed in a shader.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = MetadataOpCode, pure)]
#[allow(missing_docs)]
pub enum Metadata {
    /// The stride of an array at the given dimension.
    Stride {
        dim: Value,
        #[args(allow_ptr)]
        list: Value,
    },
    /// The shape of an array at the given dimension.
    Shape {
        dim: Value,
        #[args(allow_ptr)]
        list: Value,
    },
    /// The length of an array's underlying buffer.
    BufferLength {
        #[args(allow_ptr)]
        list: Value,
    },
}

impl Display for Metadata {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Metadata::Stride { dim, list } => write!(f, "{list}.strides[{dim}]"),
            Metadata::Shape { dim, list } => write!(f, "{list}.shape[{dim}]"),
            Metadata::BufferLength { list } => write!(f, "buffer_len({list})"),
        }
    }
}
