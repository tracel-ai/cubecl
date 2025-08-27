use alloc::{format, string::String, vec, vec::Vec};
use derive_new::new;

use super::Variable;
use crate::{OperationCode, OperationReflect};
use crate::{StorageType, TypeHash};
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
#[derive(new, Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub struct Matrix {
    pub ident: MatrixIdent,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub storage: StorageType,
    pub layout: MatrixLayout,
}

impl Matrix {
    /// Number of elements in terms of the physical storage type, accounting for packed elements
    pub fn num_elems(&self) -> u32 {
        let elems = match self.ident {
            MatrixIdent::A => self.m * self.k,
            MatrixIdent::B => self.k * self.n,
            MatrixIdent::Accumulator => self.m * self.n,
        };
        elems / self.storage.packing_factor()
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
    /// Manual execute.
    ExecuteManual {
        matrix: Matrix,
        registers_a: Vec<Variable>,
        registers_b: Vec<Variable>,
        registers_c: Vec<Variable>,
    },
    /// Scaled manual execute.
    ExecuteScaled {
        matrix: Matrix,
        registers_a: Vec<Variable>,
        registers_b: Vec<Variable>,
        registers_c: Vec<Variable>,
        scales_a: Variable,
        scales_b: Variable,
        scales_factor: u32,
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
            | CoopMma::ExecuteScaled { .. }
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
            | CmmaOpCode::ExecuteScaled
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
                registers_a,
                registers_b,
                registers_c,
            } => {
                let frag_a = comma_separated(registers_a.iter().map(|it| format!("{it}")));
                let frag_b = comma_separated(registers_b.iter().map(|it| format!("{it}")));
                let frag_c = comma_separated(registers_c.iter().map(|it| format!("{it}")));
                write!(
                    f,
                    "execute_manual_mma(
                    matrix: {matrix:?},
                    frag_a: [{frag_a}],
                    frag_b: [{frag_b}],
                    frag_c: [{frag_c}],
                )"
                )
            }
            CoopMma::ExecuteScaled {
                matrix,
                registers_a,
                registers_b,
                registers_c,
                scales_a,
                scales_b,
                scales_factor,
            } => {
                let frag_a = comma_separated(registers_a.iter().map(|it| format!("{it}")));
                let frag_b = comma_separated(registers_b.iter().map(|it| format!("{it}")));
                let frag_c = comma_separated(registers_c.iter().map(|it| format!("{it}")));
                write!(
                    f,
                    "execute_scaled_mma_{scales_factor}x(
                    matrix: {matrix:?},
                    frag_a: [{frag_a}],
                    frag_b: [{frag_b}],
                    frag_c: [{frag_c}],
                    scales_a: {scales_a},
                    scales_b: {scales_b}
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
