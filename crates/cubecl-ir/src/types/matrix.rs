use cubecl_macros_internal::TypeHash;
use derive_more::Display;
use pliron::derive::{format, pliron_type, type_interface_impl};

use crate::{
    interfaces::{AlignedType, TypedExt},
    pliron::prelude::*,
};

#[allow(missing_docs)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[pliron_type(
    name = "cube.matrix",
    format = "`matrix` $ident `<` $scope `, ` $elem_ty `, ` $shape `, ` $layout `>`",
    generate_get = true,
    verifier = "succ"
)]
pub struct MatrixType {
    pub ident: MatrixIdent,
    pub shape: MatrixShape,
    pub elem_ty: Ptr<TypeObj>,
    pub layout: MatrixLayout,
    pub scope: MatrixScope,
}

#[type_interface_impl]
impl AlignedType for MatrixType {
    fn align(&self, ctx: &Context) -> usize {
        self.elem_ty.align(ctx)
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[format("$m `x` $n `x` $k")]
pub struct MatrixShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl From<(usize, usize, usize)> for MatrixShape {
    fn from((m, n, k): (usize, usize, usize)) -> Self {
        Self { m, n, k }
    }
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[format]
#[allow(missing_docs)]
pub enum MatrixIdent {
    #[display("IdentA")]
    A,
    #[display("IdentB")]
    B,
    #[display("IdentAcc")]
    #[format("`Acc`")]
    Accumulator,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[display(rename_all = "snake_case")]
#[format]
#[allow(missing_docs)]
pub enum MatrixLayout {
    #[format("`col_major`")]
    ColMajor,
    #[format("`row_major`")]
    RowMajor,
    #[format("`undefined`")]
    Undefined,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, TypeHash, PartialEq, Eq, Hash, PartialOrd, Ord, Display)]
#[display(rename_all = "snake_case")]
#[format]
#[allow(missing_docs)]
pub enum MatrixScope {
    #[format("`plane`")]
    Plane,
    #[format("`cube`")]
    Cube,
}
