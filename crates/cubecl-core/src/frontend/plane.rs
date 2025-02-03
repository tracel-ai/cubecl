use cubecl_ir::ExpandElement;

use super::{CubePrimitive, Line};
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

/// Perform an inclusive sum operation across all units in a plane.
/// This sums all values to the "left" of the unit, including this unit's value.
/// Also known as "prefix sum" or "inclusive scan".
///
/// # Example
/// `inclusive_sum([1, 2, 3, 4, 5]) == [1, 3, 6, 10, 15]`
#[allow(unused_variables)]
pub fn plane_inclusive_sum<E: CubePrimitive>(value: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_inclusive_sum()].
pub mod plane_inclusive_sum {
    use super::*;

    /// Expand method of [plane_inclusive_sum()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(
            Plane::InclusiveSum(UnaryOperator { input }),
            out,
        ));

        output.into()
    }
}

/// Perform an exclusive sum operation across all units in a plane.
/// This sums all values to the "left" of the unit, excluding this unit's value. The 0th unit will
/// be set to `E::zero()`.
/// Also known as "exclusive prefix sum" or "exclusive scan".
///
/// # Example
/// `exclusive_sum([1, 2, 3, 4, 5]) == [0, 1, 3, 6, 10]`
#[allow(unused_variables)]
pub fn plane_exclusive_sum<E: CubePrimitive>(value: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_exclusive_sum()].
pub mod plane_exclusive_sum {
    use super::*;

    /// Expand method of [plane_exclusive_sum()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(
            Plane::ExclusiveSum(UnaryOperator { input }),
            out,
        ));

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

/// Perform an inclusive product operation across all units in a plane.
/// This multiplies all values to the "left" of the unit, including this unit's value.
/// Also known as "prefix product" or "inclusive scan".
///
/// # Example
/// `exclusive_prod([1, 2, 3, 4, 5]) == [1, 2, 6, 24, 120]`
#[allow(unused_variables)]
pub fn plane_inclusive_prod<E: CubePrimitive>(value: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_inclusive_prod()].
pub mod plane_inclusive_prod {
    use super::*;

    /// Expand method of [plane_inclusive_prod()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(
            Plane::InclusiveProd(UnaryOperator { input }),
            out,
        ));

        output.into()
    }
}

/// Perform an exclusive product operation across all units in a plane.
/// This multiplies all values to the "left" of the unit, excluding this unit's value. The 0th unit
/// will be set to `E::one()`.
/// Also known as "exclusive prefix product" or "exclusive scan".
///
/// # Example
/// `exclusive_prod([1, 2, 3, 4, 5]) == [1, 1, 2, 6, 24]`
#[allow(unused_variables)]
pub fn plane_exclusive_prod<E: CubePrimitive>(value: E) -> E {
    unexpanded!()
}

/// Module containing the expand function for [plane_exclusive_prod()].
pub mod plane_exclusive_prod {
    use super::*;

    /// Expand method of [plane_exclusive_prod()].
    pub fn expand<E: CubePrimitive>(
        scope: &mut Scope,
        elem: ExpandElementTyped<E>,
    ) -> ExpandElementTyped<E> {
        let elem: ExpandElement = elem.into();
        let output = scope.create_local(elem.item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(
            Plane::ExclusiveProd(UnaryOperator { input }),
            out,
        ));

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

/// Perform a ballot operation across all units in a plane.
/// Returns a set of 32-bit bitfields as a [`Line`], with each element containing the value from 32
/// invocations.
/// Note that line size will always be set to 4 even for `PLANE_DIM <= 64`, because we can't
/// retrieve the actual plane size at expand time. Use the runtime`PLANE_DIM` to index appropriately.
pub fn plane_ballot(_elem: bool) -> Line<u32> {
    unexpanded!()
}

/// Module containing the expand function for [plane_ballot()].
pub mod plane_ballot {

    use std::num::NonZero;

    use cubecl_ir::UIntKind;

    use super::*;

    /// Expand method of [plane_ballot()].
    pub fn expand(
        scope: &mut Scope,
        elem: ExpandElementTyped<bool>,
    ) -> ExpandElementTyped<Line<u32>> {
        let elem: ExpandElement = elem.into();
        let out_item = Item::vectorized(Elem::UInt(UIntKind::U32), NonZero::new(4));
        let output = scope.create_local(out_item);

        let out = *output;
        let input = *elem;

        scope.register(Instruction::new(
            Plane::Ballot(UnaryOperator { input }),
            out,
        ));

        output.into()
    }
}
