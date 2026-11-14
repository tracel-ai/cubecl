//! Which transports a manual-mma tile loads through ([`MmaIo`]).

use cubecl::cmma::MatrixIdent;

/// Which transports a manual-mma tile may load through; every unit stores its own cells, since
/// `stmatrix` does not reach a tile's memory window. `LoadMatrix` lets an operand
/// load through `ldmatrix` wherever the device offers it for the element and the stage serves it,
/// which the tile decides as it expands; `Manual` keeps every unit loading its own cells.
/// [`Default`] is what a caller with no reason to choose takes: `ldmatrix` for both operands where
/// it serves.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaIo {
    pub lhs_load_method: LoadMethod,
    pub rhs_load_method: LoadMethod,
    pub acc_load_method: LoadMethod,
    /// The order the `B` fragment reads its `k` rows in, to meet its `A`.
    pub rhs_keys: KeyOrder,
}

/// The order a `B` fragment reads its `k` rows in. A run of 8-bit accumulators reread as the `A`
/// operand ([`accumulators_as_a`](crate::tile::kind::plane::mma::accumulators_as_a)) holds its
/// columns where its rows lie, not where the instruction puts each `k`: the `B` reads, for each
/// `k`, the row of the column the `A` holds there, and the sum over `k` is the same.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum KeyOrder {
    /// Each `k` row where it lies.
    AsLaid,
    /// The rows a run of 8-bit accumulators reread as the `A` holds at each `k`.
    AccumulatorRun,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadMethod {
    Manual,
    LoadMatrix,
}

/// Both operands through `ldmatrix` where the device and the stage let them, the accumulator each
/// unit's own.
impl Default for MmaIo {
    fn default() -> Self {
        Self {
            lhs_load_method: LoadMethod::LoadMatrix,
            rhs_load_method: LoadMethod::LoadMatrix,
            ..Self::manual()
        }
    }
}

impl MmaIo {
    /// A config forcing the manual path for every role: what a test of that path asks for.
    pub fn manual() -> Self {
        Self {
            lhs_load_method: LoadMethod::Manual,
            rhs_load_method: LoadMethod::Manual,
            acc_load_method: LoadMethod::Manual,
            rhs_keys: KeyOrder::AsLaid,
        }
    }

    /// The transport `ident` may load through.
    pub fn load_method(&self, ident: MatrixIdent) -> LoadMethod {
        match ident {
            MatrixIdent::A => self.lhs_load_method,
            MatrixIdent::B => self.rhs_load_method,
            MatrixIdent::Accumulator => self.acc_load_method,
        }
    }
}
