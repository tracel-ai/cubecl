use crate::components::global::memory::ViewDirection;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
/// Identifier for all three tensors in a matmul
///
/// Useful to specialize some functions depending on the tensor
pub enum MatmulIdent {
    Lhs,
    Rhs,
    Out,
}

impl MatmulIdent {
    /// Equivalent to into, but type inference works better within Cube functions
    pub fn into_stage(self) -> StageIdent {
        self.into()
    }

    pub fn view_direction(&self) -> ViewDirection {
        match self {
            MatmulIdent::Lhs => ViewDirection::Col,
            MatmulIdent::Rhs => ViewDirection::Row,
            MatmulIdent::Out => ViewDirection::None,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum StageIdent {
    Lhs,
    Rhs,
    Acc,
    Out,
}

impl From<MatmulIdent> for StageIdent {
    fn from(matmul_ident: MatmulIdent) -> Self {
        match matmul_ident {
            MatmulIdent::Lhs => StageIdent::Lhs,
            MatmulIdent::Rhs => StageIdent::Rhs,
            MatmulIdent::Out => StageIdent::Acc,
        }
    }
}
