use super::{CubeContext, CubePrimitive, ExpandElement};
use crate::{
    ir::{Elem, InitOperator, Item, Operation, Subcube, UnaryOperator},
    unexpanded,
};
use crate::{new_ir::Primitive, prelude::ExpandElementTyped};

/// Returns true if the cube unit has the lowest subcube_unit_id among active unit in the subcube
pub fn subcube_elect() -> bool {
    unexpanded!()
}

pub mod subcube_elect {
    use crate::new_ir::{Expr, SubcubeElectExpr};

    pub fn expand() -> impl Expr<Output = bool> {
        SubcubeElectExpr
    }
}

pub fn subcube_elect_expand<E: CubePrimitive>(context: &mut CubeContext) -> ExpandElement {
    let output = context.create_local(Item::new(Elem::Bool));

    let out = *output;

    context.register(Operation::Subcube(Subcube::Elect(InitOperator { out })));

    output
}

/// Perform a reduce sum operation across all units in a subcube.
#[allow(unused_variables)]
pub fn subcube_sum<E: Primitive>(value: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [subcube_sum()].
pub mod subcube_sum {
    use crate::new_ir::{Expr, SubcubeSumExpr};

    use super::*;

    /// Expand method of [subcube_sum()].
    pub fn __expand<E: CubePrimitive>(
        context: &mut CubeContext,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = context.create_local(elem.item());

        let out = *output;
        let input = *elem;

        context.register(Operation::Subcube(Subcube::Sum(UnaryOperator {
            input,
            out,
        })));

        output.into()
    }

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
    use crate::new_ir::{Expr, SubcubeProdExpr};

    use super::*;

    /// Expand method of [subcube_prod()].
    pub fn __expand<E: CubePrimitive>(
        context: &mut CubeContext,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = context.create_local(elem.item());

        let out = *output;
        let input = *elem;

        context.register(Operation::Subcube(Subcube::Prod(UnaryOperator {
            input,
            out,
        })));

        output.into()
    }

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
    use crate::new_ir::{Expr, SubcubeMaxExpr};

    use super::*;

    /// Expand method of [subcube_max()].
    pub fn __expand<E: CubePrimitive>(
        context: &mut CubeContext,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = context.create_local(elem.item());

        let out = *output;
        let input = *elem;

        context.register(Operation::Subcube(Subcube::Max(UnaryOperator {
            input,
            out,
        })));

        output.into()
    }

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
    use crate::new_ir::{Expr, SubcubeMinExpr};

    use super::*;

    /// Expand method of [subcube_min()].
    pub fn __expand<E: CubePrimitive>(
        context: &mut CubeContext,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = context.create_local(elem.item());

        let out = *output;
        let input = *elem;

        context.register(Operation::Subcube(Subcube::Min(UnaryOperator {
            input,
            out,
        })));

        output.into()
    }

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
    use crate::{
        new_ir::{Expr, SubcubeAllExpr},
        prelude::Bool,
    };

    /// Expand method of [subcube_all()].
    pub fn __expand(
        context: &mut CubeContext,
        elem: ExpandElementTyped<Bool>,
    ) -> ExpandElementTyped<Bool> {
        let elem: ExpandElement = elem.into();
        let output = context.create_local(elem.item());

        let out = *output;
        let input = *elem;

        context.register(Operation::Subcube(Subcube::All(UnaryOperator {
            input,
            out,
        })));

        output.into()
    }

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
    use crate::new_ir::{Expr, SubcubeAnyExpr};

    pub fn expand(elem: impl Expr<Output = bool>) -> impl Expr<Output = bool> {
        SubcubeAnyExpr::new(elem)
    }
}

pub fn subcube_broadcast<E: Primitive>(_value: E, _index: u32) -> E {
    unexpanded!()
}

pub mod subcube_broadcast {
    use crate::new_ir::{BinaryOp, Expr, Primitive, SubcubeBroadcastExpr};

    pub fn expand<E: Primitive>(
        value: impl Expr<Output = E>,
        index: impl Expr<Output = u32>,
    ) -> impl Expr<Output = E> {
        SubcubeBroadcastExpr(BinaryOp::new(value, index))
    }
}
