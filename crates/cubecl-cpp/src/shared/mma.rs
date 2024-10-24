use std::fmt::Display;

use super::{Dialect, Elem, Variable};

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum FragmentIdent {
    A,
    B,
    Accumulator,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum FragmentLayout {
    ColMajor,
    RowMajor,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub struct Fragment<D: Dialect> {
    pub ident: FragmentIdent,
    pub m: u8,
    pub n: u8,
    pub k: u8,
    pub elem: Elem<D>,
    pub layout: Option<FragmentLayout>,
}

/// Warp Matrix-Multiply and Accumulate Instruction.
#[derive(Debug, Clone, Copy)]
pub enum WmmaInstruction<D: Dialect> {
    /// Fill the fragment with the value.
    Fill {
        frag: Variable<D>,
        value: Variable<D>,
    },
    /// Load the value into the fragment given the stride.
    Load {
        frag: Variable<D>,
        value: Variable<D>,
        stride: Variable<D>,
        layout: Option<FragmentLayout>,
    },
    /// Executes D=A*B+C;
    ///
    /// For implementing a matmul, `D=C` : `C+=A*B`
    Execute {
        frag_a: Variable<D>,
        frag_b: Variable<D>,
        frag_c: Variable<D>,
        frag_d: Variable<D>,
    },
    /// Store the fragment in an output variable following the stride and the layout.
    Store {
        output: Variable<D>,
        frag: Variable<D>,
        stride: Variable<D>,
        layout: FragmentLayout,
    },
}

impl Display for FragmentLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentLayout::ColMajor => f.write_str("nvcuda::wmma::col_major"),
            FragmentLayout::RowMajor => f.write_str("nvcuda::wmma::row_major"),
        }
    }
}

impl Display for FragmentIdent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentIdent::A => f.write_str("nvcuda::wmma::matrix_a"),
            FragmentIdent::B => f.write_str("nvcuda::wmma::matrix_b"),
            FragmentIdent::Accumulator => f.write_str("nvcuda::wmma::accumulator"),
        }
    }
}

impl<D: Dialect> Display for Fragment<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.layout {
            Some(layout) => write!(
                f,
                "nvcuda::wmma::fragment<{}, {}, {}, {}, {}, {}>",
                self.ident, self.m, self.n, self.k, self.elem, layout
            ),
            None => write!(
                f,
                "nvcuda::wmma::fragment<{}, {}, {}, {}, {}>",
                self.ident, self.m, self.n, self.k, self.elem,
            ),
        }
    }
}

impl<D: Dialect> Display for WmmaInstruction<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WmmaInstruction::Fill { frag, value } => {
                writeln!(f, "nvcuda::wmma::fill_fragment({frag}, {value});")
            }
            WmmaInstruction::Load {
                frag,
                value,
                stride,
                layout: None,
            } => writeln!(
                f,
                "nvcuda::wmma::load_matrix_sync({frag}, {value}, {stride});"
            ),
            WmmaInstruction::Load {
                frag,
                value,
                stride,
                layout: Some(layout),
            } => {
                let layout = match layout {
                    FragmentLayout::ColMajor => "nvcuda::wmma::mem_col_major",
                    FragmentLayout::RowMajor => "nvcuda::wmma::mem_row_major",
                };
                writeln!(
                    f,
                    "nvcuda::wmma::load_matrix_sync({frag}, {value}, {stride}, {layout});"
                )
            }
            WmmaInstruction::Execute {
                frag_a,
                frag_b,
                frag_c,
                frag_d,
            } => writeln!(
                f,
                "nvcuda::wmma::mma_sync({frag_d}, {frag_a}, {frag_b}, {frag_c});"
            ),
            WmmaInstruction::Store {
                output,
                frag,
                stride,
                layout,
            } => {
                let layout = match layout {
                    FragmentLayout::ColMajor => "nvcuda::wmma::mem_col_major",
                    FragmentLayout::RowMajor => "nvcuda::wmma::mem_row_major",
                };

                writeln!(
                    f,
                    "nvcuda::wmma::store_matrix_sync({output}, {frag}, {stride}, {layout});"
                )
            }
        }
    }
}
