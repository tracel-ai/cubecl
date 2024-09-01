use crate::new_ir::Expr;
use crate::prelude::Primitive;
use crate::unexpanded;

/// Returns true if the cube unit has the lowest subcube_unit_id among active unit in the subcube
pub fn subcube_elect() -> bool {
    unexpanded!()
}

pub mod subcube_elect {
    use super::*;
    use crate::new_ir::SubcubeElectExpr;

    pub fn expand() -> impl Expr<Output = bool> {
        SubcubeElectExpr
    }
}

/// Perform a reduce sum operation across all units in a subcube.
#[allow(unused_variables)]
pub fn subcube_sum<E: Primitive>(value: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [subcube_sum()].
pub mod subcube_sum {
    use super::*;
    use crate::new_ir::SubcubeSumExpr;

    pub fn expand<E: Primitive>(elem: impl Expr<Output = E>) -> impl Expr<Output = E> {
        SubcubeSumExpr::new(elem)
    }
}

/// Perform a reduce prod operation across all units in a subcube.
pub fn subcube_prod<E: Primitive>(_elem: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [subcube_prod()].
pub mod subcube_prod {
    use super::*;
    use crate::new_ir::SubcubeProdExpr;

    pub fn expand<E: Primitive>(elem: impl Expr<Output = E>) -> impl Expr<Output = E> {
        SubcubeProdExpr::new(elem)
    }
}

/// Perform a reduce max operation across all units in a subcube.
pub fn subcube_max<E: Primitive>(_elem: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [subcube_max()].
pub mod subcube_max {
    use super::*;
    use crate::new_ir::SubcubeMaxExpr;

    pub fn expand<E: Primitive>(elem: impl Expr<Output = E>) -> impl Expr<Output = E> {
        SubcubeMaxExpr::new(elem)
    }
}

/// Perform a reduce min operation across all units in a subcube.
pub fn subcube_min<E: Primitive>(_elem: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [subcube_min()].
pub mod subcube_min {
    use super::*;
    use crate::new_ir::SubcubeMinExpr;

    pub fn expand<E: Primitive>(elem: impl Expr<Output = E>) -> impl Expr<Output = E> {
        SubcubeMinExpr::new(elem)
    }
}

/// Perform a reduce all operation across all units in a subcube.
pub fn subcube_all(_elem: bool) -> bool {
    unexpanded!()
}

/// Module containing the expand function for [subcube_all()].
pub mod subcube_all {
    use super::*;
    use crate::new_ir::SubcubeAllExpr;

    pub fn expand(elem: impl Expr<Output = bool>) -> impl Expr<Output = bool> {
        SubcubeAllExpr::new(elem)
    }
}

/// Perform a reduce all operation across all units in a subcube.
pub fn subcube_any(_elem: bool) -> bool {
    unexpanded!()
}

/// Module containing the expand function for [subcube_all()].
pub mod subcube_any {
    use super::*;
    use crate::new_ir::SubcubeAnyExpr;

    pub fn expand(elem: impl Expr<Output = bool>) -> impl Expr<Output = bool> {
        SubcubeAnyExpr::new(elem)
    }
}

pub fn subcube_broadcast<E: Primitive>(_value: E, _index: u32) -> E {
    unexpanded!()
}

pub mod subcube_broadcast {
    use super::*;
    use crate::new_ir::{BinaryOp, Expr, SubcubeBroadcastExpr};

    pub fn expand<E: Primitive>(
        value: impl Expr<Output = E>,
        index: impl Expr<Output = u32>,
    ) -> impl Expr<Output = E> {
        SubcubeBroadcastExpr(BinaryOp::new(value, index))
    }
}
