use cubecl_ir::ExpandElement;

use super::CubePrimitive;
use crate::prelude::ExpandElementTyped;
use crate::{
    ir::{Elem, Instruction, Item, Plane, Scope, UnaryOperator},
    unexpanded,
};

/// Returns true if the cube unit has the lowest plane_unit_id among active unit in the plane
pub fn plane_elect() -> bool {
    unexpanded!()
}

/// Module containing the expand function for [plane_elect()].
pub mod plane_elect {

    use super::*;

    /// Expand method of [plane_elect()].
    pub fn expand(scope: &mut Scope) -> ExpandElementTyped<bool> {
        let output = scope.create_local(Item::new(Elem::Bool));
        let out = *output;

        scope.register(Instruction::new(Plane::Elect, out));

        output.into()
    }
}

/// Broadcasts the value from the specified plane unit at the given index
/// to all active units within that plane.
#[allow(unused_variables)]
pub fn plane_broadcast<E: CubePrimitive>(value: E, index: u32) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_broadcast()].
pub mod plane_broadcast {

    use super::*;

    /// Expand method of [plane_broadcast()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        value: ExpandElementTyped<E>,
        id: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<E> {
        let output = scope.create_local(value.expand.item);
        let out = *output;
        let lhs = *value.expand;
        let rhs = *id.expand;

        scope.register(Instruction::new(
            Plane::Broadcast(crate::ir::BinaryOperator { lhs, rhs }),
            out,
        ));

        output.into()
    }
}

/// Perform a reduce sum operation across all units in a plane.
#[allow(unused_variables)]
pub fn plane_sum<E: CubePrimitive>(value: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_sum()].
pub mod plane_sum {
    use super::*;

    /// Expand method of [plane_sum()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(Plane::Sum(UnaryOperator { input }), out));

        output.into()
    }
}

/// Perform a reduce prod operation across all units in a plane.
pub fn plane_prod<E: CubePrimitive>(_elem: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_prod()].
pub mod plane_prod {
    use super::*;

    /// Expand method of [plane_prod()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(Plane::Prod(UnaryOperator { input }), out));

        output.into()
    }
}

/// Perform a reduce max operation across all units in a plane.
pub fn plane_max<E: CubePrimitive>(_elem: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_max()].
pub mod plane_max {
    use super::*;

    /// Expand method of [plane_max()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(Plane::Max(UnaryOperator { input }), out));

        output.into()
    }
}

/// Perform a reduce min operation across all units in a plane.
pub fn plane_min<E: CubePrimitive>(_elem: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_min()].
pub mod plane_min {
    use super::*;

    /// Expand method of [plane_min()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(Plane::Min(UnaryOperator { input }), out));

        output.into()
    }
}

/// Perform a reduce all operation across all units in a plane.
pub fn plane_all(_elem: bool) -> bool {
    unexpanded!()
}

/// Module containing the expand function for [plane_all()].
pub mod plane_all {

    use super::*;

    /// Expand method of [plane_all()].
    pub fn expand(scope: &mut Scope, elem: ExpandElementTyped<bool>) -> ExpandElementTyped<bool> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(Plane::All(UnaryOperator { input }), out));

        output.into()
    }
}

/// Perform a reduce any operation across all units in a plane.
pub fn plane_any(_elem: bool) -> bool {
    unexpanded!()
}

/// Module containing the expand function for [plane_any()].
pub mod plane_any {

    use super::*;

    /// Expand method of [plane_any()].
    pub fn expand(scope: &mut Scope, elem: ExpandElementTyped<bool>) -> ExpandElementTyped<bool> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(Plane::Any(UnaryOperator { input }), out));

        output.into()
    }
}
