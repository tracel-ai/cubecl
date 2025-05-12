use alloc::{format, string::String, vec, vec::Vec};

use super::{Elem, Variable};
use crate::TypeHash;
use crate::{OperationCode, OperationReflect};
use core::fmt::Display;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum MatrixIdent {
    A,
    B,
    Accumulator,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub enum MatrixLayout {
    ColMajor,
    RowMajor,
    Undefined,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub struct Matrix {
    pub ident: MatrixIdent,
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub elem: Elem,
    pub layout: MatrixLayout,
}

/// Cooperative Matrix-Multiply and Accumulate Instruction.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationCode)]
#[operation(opcode_name = CmmaOpCode)]
#[allow(missing_docs)]
pub enum CoopMma {
    /// Fill the matrix with the value.
    Fill { value: Variable },
    /// Load the value into the matrix given the stride.
    Load {
        value: Variable,
        stride: Variable,
        offset: Variable,
        layout: Option<MatrixLayout>,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        mat_a: Variable,
        mat_b: Variable,
        mat_c: Variable,
    },
    /// Store the matrix in an output variable following the stride and the layout.
    Store {
        mat: Variable,
        stride: Variable,
        offset: Variable,
        layout: MatrixLayout,
    },
    /// Cast a fragment to another type.
    Cast { input: Variable },
}

impl OperationReflect for CoopMma {
    type OpCode = CmmaOpCode;

    fn op_code(&self) -> Self::OpCode {
        self.__match_opcode()
    }

    fn args(&self) -> Option<Vec<Variable>> {
        match self {
            CoopMma::Fill { value } => Some(vec![*value]),
            CoopMma::Load { .. } | CoopMma::Execute { .. } | CoopMma::Store { .. } => None,
            CoopMma::Cast { input } => Some(vec![*input]),
        }
    }

    fn from_code_and_args(op_code: Self::OpCode, args: &[Variable]) -> Option<Self> {
        match op_code {
            CmmaOpCode::Fill => Some(CoopMma::Fill { value: args[0] }),
            CmmaOpCode::Load | CmmaOpCode::Execute | CmmaOpCode::Store => None,
            CmmaOpCode::Cast => Some(CoopMma::Cast { input: args[0] }),
        }
    }
}

impl Display for CoopMma {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoopMma::Fill { value } => write!(f, "{value}"),
            CoopMma::Load {
                value,
                stride,
                offset,
                layout,
            } => {
                let layout = layout
                    .map(|it| format!(", layout: {it:?}"))
                    .unwrap_or(String::new());
                write!(
                    f,
                    "matrix_load({value}, stride: {stride}{layout}, offset: {offset})"
                )
            }
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
            } => write!(f, "execute_cmma({mat_a}, {mat_b}, {mat_c})"),
            CoopMma::Store {
                mat,
                stride,
                offset,
                layout,
            } => write!(
                f,
                "matrix_store({}, stride: {}, layout: {:?}, offset: {:?})",
                mat, stride, layout, offset
            ),
            CoopMma::Cast { input } => {
                write!(f, "matrix_cast(input: {input})")
            }
        }
    }
}
