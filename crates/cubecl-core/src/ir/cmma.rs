use std::fmt::Display;

use serde::{Deserialize, Serialize};

use super::{Elem, Variable};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum MatrixIdent {
    A,
    B,
    Accumulator,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum MatrixLayout {
    ColMajor,
    RowMajor,
    Undefined,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
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
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[allow(missing_docs)]
pub enum CoopMma {
    /// Fill the matrix with the value.
    Fill { mat: Variable, value: Variable },
    /// Load the value into the matrix given the stride.
    Load {
        mat: Variable,
        value: Variable,
        stride: Variable,
        layout: Option<MatrixLayout>,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        mat_a: Variable,
        mat_b: Variable,
        mat_c: Variable,
        mat_d: Variable,
    },
    /// Store the matrix in an output variable following the stride and the layout.
    Store {
        output: Variable,
        mat: Variable,
        stride: Variable,
        layout: MatrixLayout,
    },
}

impl Display for CoopMma {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CoopMma::Fill { mat, value } => write!(f, "matrix_fill({}, {})", mat, value),
            CoopMma::Load {
                mat,
                value,
                stride,
                layout,
            } => {
                let layout = layout
                    .map(|it| format!(", layout: {it:?}"))
                    .unwrap_or(String::new());
                write!(
                    f,
                    "matrix_load({}, {}, stride: {}{layout})",
                    mat, value, stride
                )
            }
            CoopMma::Execute {
                mat_a,
                mat_b,
                mat_c,
                mat_d,
            } => write!(
                f,
                "{} = execute_cmma({}, {}, {})",
                mat_d, mat_a, mat_b, mat_c
            ),
            CoopMma::Store {
                output,
                mat,
                stride,
                layout,
            } => write!(
                f,
                "matrix_store({}, {}, stride: {}, layout: {:?})",
                mat, output, stride, layout
            ),
        }
    }
}
