use alloc::{format, string::String};
use derive_more::Display;
use derive_new::new;

use super::Value;
use crate::{Closure, OperationReflect};
use crate::{StorageType, TypeHash};
use core::fmt::Display;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[allow(missing_docs)]
pub enum MatrixIdent {
    #[display("IdentA")]
    A,
    #[display("IdentB")]
    B,
    #[display("IdentAcc")]
    Accumulator,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[display(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum MatrixLayout {
    ColMajor,
    RowMajor,
    Undefined,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[display(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum MatrixScope {
    Plane,
    Cube,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(missing_docs)]
pub struct MatrixType {
    pub ident: MatrixIdent,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub storage: StorageType,
    pub layout: MatrixLayout,
    pub scope: MatrixScope,
}

impl MatrixType {
    /// Number of elements in terms of the physical storage type, accounting for packed elements
    pub fn num_elems(&self) -> usize {
        let elems = match self.ident {
            MatrixIdent::A => self.m * self.k,
            MatrixIdent::B => self.k * self.n,
            MatrixIdent::Accumulator => self.m * self.n,
        };
        elems / self.storage.packing_factor()
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClampMode {
    Undefined,
    Constant(u32),
    ClampToEdge,
    Repeat,
    RepeatMirrored,
}

/// Cooperative Matrix-Multiply and Accumulate Instruction.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, TypeHash, PartialEq, Eq, Hash, OperationReflect)]
#[operation(opcode_name = CmmaOpCode)]
#[allow(missing_docs)]
pub enum CoopMma {
    /// Fill the matrix with the value.
    Fill { value: Value },
    /// Load the value into the matrix given the stride.
    Load {
        #[args(allow_ptr, ptr_read)]
        ptr: Value,
        stride: Value,
        #[args(skip)]
        layout: Option<MatrixLayout>,
    },
    /// Load the value into the matrix given the tensor layout.
    LoadTensor {
        #[args(allow_ptr, ptr_read)]
        buffer: Value,
        layout: Value,
        #[args(skip)]
        view: Option<Value>,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        mat_a: Value,
        mat_b: Value,
        mat_c: Value,
    },
    /// Store the matrix in an output variable following the stride and the layout.
    Store {
        mat: Value,
        stride: Value,
        #[args(allow_ptr, ptr_write)]
        destination: Value,
        #[args(skip)]
        layout: MatrixLayout,
    },
    /// Store the matrix in an output variable following the tensor layout.
    StoreTensor {
        mat: Value,
        layout: Value,
        #[args(skip)]
        view: Option<Value>,
    },
    /// Cast a fragment to another type.
    Cast { input: Value },

    /// Row index of nth element in the lane
    RowIndex {
        lane_id: Value,
        i: Value,
        #[args(skip)]
        matrix: MatrixType,
    },
    /// Column index of nth element in the lane
    ColIndex {
        lane_id: Value,
        i: Value,
        #[args(skip)]
        matrix: MatrixType,
    },
    /// Execute a CUDA `ldmatrix` instruction
    LoadMatrix {
        #[args(allow_ptr, ptr_read)]
        ptr: Value,
        factor: usize,
        transpose: bool,
    },
    /// Execute a CUDA `stmatrix` instruction
    StoreMatrix {
        registers: Value,
        factor: usize,
        transpose: bool,
        #[args(allow_ptr, ptr_write)]
        destination: Value,
    },
    /// Manual execute.
    ExecuteManual {
        #[args(skip)]
        matrix: MatrixType,
        registers_a: Value,
        registers_b: Value,
        registers_c: Value,
    },
    /// Scaled manual execute.
    ExecuteScaled {
        #[args(skip)]
        matrix: MatrixType,
        registers_a: Value,
        registers_b: Value,
        registers_c: Value,
        scales_a: Value,
        scales_b: Value,
        scales_factor: usize,
    },
    ExecuteElementwise {
        matrix: Value,
        #[args(skip)]
        op: Closure,
    },
}

impl Display for CoopMma {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoopMma::Fill { value } => write!(f, "fill({value})"),
            CoopMma::Load {
                ptr,
                stride,
                layout,
            } => {
                let layout = layout
                    .map(|it| format!(", layout: {it:?}"))
                    .unwrap_or(String::new());
                write!(f, "matrix_load({ptr}, stride: {stride}{layout})")
            }
            CoopMma::LoadTensor {
                buffer,
                layout,
                view,
            } => {
                let view = view.map(|it| format!(", view: {it}")).unwrap_or_default();
                write!(f, "matrix_load_tensor({buffer}, layout: {layout}{view})")
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
                write!(
                    f,
                    "execute_manual_mma(
                    matrix: {matrix:?},
                    frag_a: {registers_a},
                    frag_b: {registers_b},
                    frag_c: {registers_c},
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
                write!(
                    f,
                    "execute_scaled_mma_{scales_factor}x(
                    matrix: {matrix:?},
                    frag_a: {registers_a},
                    frag_b: {registers_b},
                    frag_c: {registers_c},
                    scales_a: {scales_a},
                    scales_b: {scales_b}
                )"
                )
            }
            CoopMma::ExecuteElementwise { matrix, op } => {
                write!(f, "execute_elementwise({matrix}, {op})")
            }
            CoopMma::Store {
                mat,
                stride,
                destination,
                layout,
            } => write!(
                f,
                "matrix_store({mat}, stride: {stride}, layout: {layout:?}, dest: {destination})"
            ),
            CoopMma::StoreTensor { mat, layout, .. } => {
                write!(f, "matrix_store_tensor({mat}, layout: {layout})")
            }
            CoopMma::Cast { input } => {
                write!(f, "matrix_cast(input: {input})")
            }
            CoopMma::RowIndex { lane_id, i, matrix } => {
                write!(f, "row_idx(lane_id: {lane_id}, i: {i}, matrix: {matrix:?})",)
            }
            CoopMma::ColIndex { lane_id, i, matrix } => {
                write!(f, "col_idx(lane_id: {lane_id}, i: {i}, matrix: {matrix:?})",)
            }
            CoopMma::LoadMatrix {
                ptr,
                factor,
                transpose,
            } => {
                write!(f, "ldmatrix_{factor}x({ptr}, transpose: {transpose})")
            }
            CoopMma::StoreMatrix {
                registers,
                factor,
                transpose,
                destination,
            } => {
                write!(
                    f,
                    "stmatrix_{factor}x({registers}, dest: {destination}, transpose: {transpose})"
                )
            }
        }
    }
}
