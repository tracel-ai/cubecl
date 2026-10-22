//! Which transports a manual-mma tile loads and stores through ([`MmaIo`]).

use cubecl::{
    cmma::MatrixIdent,
    ir::{DeviceProperties, ElemType},
};

/// Which transports a manual-mma tile may load and store through. `LoadMatrix` lets an operand
/// load through `ldmatrix` wherever the device offers it for the element and the stage serves it,
/// which the tile decides as it expands; `Manual` keeps every unit loading its own cells.
/// [`Default`] is what a caller with no reason to choose takes: `ldmatrix` for both operands where
/// it serves.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MmaIo {
    pub lhs_load_method: LoadMethod,
    pub rhs_load_method: LoadMethod,
    pub acc_load_method: LoadMethod,
    pub store_method: StoreMethod,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum LoadMethod {
    Manual,
    LoadMatrix,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum StoreMethod {
    Manual,
    StoreMatrix,
}

/// Both operands through `ldmatrix` where the device and the stage let them, the accumulator and
/// the store each unit's own: `stmatrix` does not reach a tile's memory window.
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
    /// Select each role's transport from the device's `ldmatrix`/`stmatrix` support.
    pub fn new(
        device_props: &DeviceProperties,
        lhs_stage: ElemType,
        rhs_stage: ElemType,
        acc_stage: ElemType,
    ) -> Self {
        Self {
            lhs_load_method: load_method(device_props, lhs_stage),
            rhs_load_method: load_method(device_props, rhs_stage),
            acc_load_method: load_method(device_props, acc_stage),
            store_method: store_method(device_props, acc_stage),
        }
    }

    /// A config forcing the manual path for every role: what a test of that path asks for.
    pub fn manual() -> Self {
        Self {
            lhs_load_method: LoadMethod::Manual,
            rhs_load_method: LoadMethod::Manual,
            acc_load_method: LoadMethod::Manual,
            store_method: StoreMethod::Manual,
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

fn load_method(device_props: &DeviceProperties, dtype: ElemType) -> LoadMethod {
    if device_props.features.matmul.ldmatrix.contains(&dtype) {
        LoadMethod::LoadMatrix
    } else {
        LoadMethod::Manual
    }
}

fn store_method(device_props: &DeviceProperties, dtype: ElemType) -> StoreMethod {
    if device_props.features.matmul.stmatrix.contains(&dtype) {
        StoreMethod::StoreMatrix
    } else {
        StoreMethod::Manual
    }
}
