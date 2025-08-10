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
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub elem: Elem,
    pub layout: MatrixLayout,
}

impl Matrix {
    pub fn new(
        ident: MatrixIdent,
        m: u32,
        n: u32,
        k: u32,
        elem: Elem,
        layout: MatrixLayout,
    ) -> Self {
        Matrix {
            ident,
            m,
            n,
            k,
            elem,
            layout,
        }
    }

    pub fn num_elems(&self) -> u32 {
        match self.ident {
            MatrixIdent::A => self.m * self.k,
            MatrixIdent::B => self.k * self.n,
            MatrixIdent::Accumulator => self.m * self.n,
        }
    }
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

    /// Row index of nth element in the lane
    RowIndex {
        lane_id: Variable,
        i: Variable,
        matrix: Matrix,
    },
    /// Column index of nth element in the lane
    ColIndex {
        lane_id: Variable,
        i: Variable,
        matrix: Matrix,
    },
    /// Manual execute. No out because we don't have native composites, and out is multiple variables.
    ExecuteManual {
        matrix: Matrix,
        a_registers: Vec<Variable>,
        b_registers: Vec<Variable>,
        c_registers: Vec<Variable>,
        d_registers: Vec<Variable>,
    },
}

impl OperationReflect for CoopMma {
    type OpCode = CmmaOpCode;

    fn op_code(&self) -> Self::OpCode {
        self.__match_opcode()
    }

    fn args(&self) -> Option<Vec<Variable>> {
        match self {
            CoopMma::Fill { value } => Some(vec![*value]),
            CoopMma::Load { .. }
            | CoopMma::Execute { .. }
            | CoopMma::ExecuteManual { .. }
            | CoopMma::Store { .. }
            | CoopMma::RowIndex { .. }
            | CoopMma::ColIndex { .. } => None,
            CoopMma::Cast { input } => Some(vec![*input]),
        }
    }

    fn from_code_and_args(op_code: Self::OpCode, args: &[Variable]) -> Option<Self> {
        match op_code {
            CmmaOpCode::Fill => Some(CoopMma::Fill { value: args[0] }),
            CmmaOpCode::Load
            | CmmaOpCode::Execute
            | CmmaOpCode::ExecuteManual
            | CmmaOpCode::Store
            | CmmaOpCode::RowIndex
            | CmmaOpCode::ColIndex => None,
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
            CoopMma::ExecuteManual {
                matrix,
                a_registers,
                b_registers,
                c_registers,
                d_registers,
            } => {
                let frag_a = comma_separated(a_registers.iter().map(|it| format!("{it}")));
                let frag_b = comma_separated(b_registers.iter().map(|it| format!("{it}")));
                let frag_c = comma_separated(c_registers.iter().map(|it| format!("{it}")));
                let frag_d = comma_separated(d_registers.iter().map(|it| format!("{it}")));
                write!(
                    f,
                    "execute_manual_mma(
                    matrix: {matrix:?},
                    frag_a: [{frag_a}],
                    frag_b: [{frag_b}],
                    frag_c: [{frag_c}],
                    frag_d: [{frag_d}]
                )"
                )
            }
            CoopMma::Store {
                mat,
                stride,
                offset,
                layout,
            } => write!(
                f,
                "matrix_store({mat}, stride: {stride}, layout: {layout:?}, offset: {offset:?})"
            ),
            CoopMma::Cast { input } => {
                write!(f, "matrix_cast(input: {input})")
            }
            CoopMma::RowIndex { lane_id, i, matrix } => {
                write!(f, "row_idx(lane_id: {lane_id}, i: {i}, matrix: {matrix:?})",)
            }
            CoopMma::ColIndex { lane_id, i, matrix } => {
                write!(f, "col_idx(lane_id: {lane_id}, i: {i}, matrix: {matrix:?})",)
            }
        }
    }
}

fn comma_separated(it: impl IntoIterator<Item = String>) -> String {
    it.into_iter().collect::<Vec<_>>().join(", ")
}
