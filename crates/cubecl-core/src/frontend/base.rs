use cubecl_ir::Scope;
use pliron::irbuild::inserter::OpInsertionPoint;

use crate::frontend::{ConstRuntimeAssign, CubeType, NativeExpand, PlaceExpand};

#[macro_export]
macro_rules! unexpanded {
    () => ({
        panic!("Unexpanded Cube functions should not be called. ");
    });
    ($msg:expr) => ({
        panic!($msg);
    });
    ($fmt:expr, $($arg:tt)*) => ({
        panic!($fmt, $($arg)*);
    });
}

#[macro_export]
macro_rules! expand_error {
    () => ({
        panic!("An error occurred during kernel expansion");
    });
    ($msg:expr) => ({
        panic!(concat!("An error occurred during kernel expansion:\n", $msg));
    });
    ($fmt:expr, $($arg:tt)*) => ({
        panic!(concat!("An error occurred during kernel expansion:\n", $fmt), $($arg)*);
    });
}

#[macro_export]
macro_rules! expand_assert {
    ($cond:expr) => ({
        assert!($cond, "An error occurred during kernel expansion");
    });
    ($cond:expr, $msg:expr) => ({
        assert!($cond, concat!("An error occurred during kernel expansion:\n", $msg));
    });
    ($cond:expr, $fmt:expr, $($arg:tt)*) => ({
        assert!($cond, concat!("An error occurred during kernel expansion:\n", $fmt), $($arg)*);
    });
}

/// `let size!(N) = size;` defines a size `N` to the end of its block. A function re-entered on
/// another size gives `N` back to its outer call when the inner one returns.
#[macro_export]
macro_rules! size {
    ($name: ident) => {
        $name
    };
}

/// `let define!(T) = elem;` defines a type `T` naming `elem` to the end of its block. A
/// function re-entered on another element gives `T` back to its outer call when the inner one
/// returns.
#[macro_export]
macro_rules! define {
    ($name: ident) => {
        $name
    };
}

pub fn unexpanded_value<T>() -> T {
    unexpanded!()
}

/// Used for `let x;` local bindings. These are technically immutable in Rust, but because they can
/// be visible at a higher level scope than their assignment we need to allocate a slot of them at
/// that level and treat them as mutable.
pub fn uninit_local<T: CubeType>(scope: &Scope) -> NativeExpand<T>
where
    T::ExpandType: ConstRuntimeAssign<Expand = NativeExpand<T>>,
{
    T::ExpandType::const_init_mut(scope)
}

#[doc(hidden)]
pub struct PlaceHelper<T>(pub T);

#[doc(hidden)]
pub trait PlainPlaceAccess<T> {
    fn __expand_read_place(self, scope: &Scope) -> T;
}

#[doc(hidden)]
impl<T> PlainPlaceAccess<T> for PlaceHelper<T> {
    fn __expand_read_place(self, _: &Scope) -> T {
        self.0
    }
}

#[doc(hidden)]
impl<T: PlaceExpand> PlaceHelper<T> {
    pub fn __expand_read_place(self, scope: &Scope) -> T {
        self.0.read_place(scope)
    }
}

/// Receivers must be loaded before their arguments are evaluated, but the call site doesn't know
/// the receiver type. So methods need to place their receiver load "out of order", before the args
/// are evaluated. This works in tandem with `__expand_read_receiver` to temporarily override the
/// insertion point for the receiver read to the point before the method args start evaluating.
///
/// The macro needs to generate `scope.start_call_args()` in a block with the scope arg, so it's
/// evaluated right before the first actual arg.
pub fn __expand_method_call<R>(scope: &Scope, call: impl FnOnce(&Scope) -> R) -> R {
    let res = call(scope);
    scope.end_call_args();
    res
}

pub fn __expand_read_receiver<T: PlaceExpand>(scope: &Scope, value: T) -> T {
    let current_pos = scope.inserter().get_insertion_point();
    let before_recv = scope.before_call_args_pos();

    // If the point is the same we need to append normally and not reset, otherwise the point gets
    // set before the load and everything breaks
    if is_same_point(&current_pos, &before_recv) {
        value.read_place(scope)
    } else {
        scope.inserter().set_insertion_point(before_recv);
        let value = value.read_place(scope);
        scope.inserter().set_insertion_point(current_pos);
        value
    }
}

// Can't be bothered to upstream an equality impl
fn is_same_point(lhs: &OpInsertionPoint, rhs: &OpInsertionPoint) -> bool {
    match (lhs, rhs) {
        (OpInsertionPoint::Unset, OpInsertionPoint::Unset) => true,
        (OpInsertionPoint::AtBlockStart(lhs), OpInsertionPoint::AtBlockStart(rhs)) => lhs == rhs,
        (OpInsertionPoint::AtBlockEnd(lhs), OpInsertionPoint::AtBlockEnd(rhs)) => lhs == rhs,
        (OpInsertionPoint::AfterOperation(lhs), OpInsertionPoint::AfterOperation(rhs)) => {
            lhs == rhs
        }
        (OpInsertionPoint::BeforeOperation(lhs), OpInsertionPoint::BeforeOperation(rhs)) => {
            lhs == rhs
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use crate as cubecl;
    use crate::prelude::*;
    use cubecl_ir::{
        AddressType, FloatKind,
        settings::{Dim3, ExecutionMode, KernelSettings},
    };

    /// What `T` names after the call one depth down has named it again: `f16` at the bottom,
    /// `f32` above it.
    #[cube]
    fn named_after_reentry(#[comptime] depth: u32) -> comptime_type!(ElemType) {
        let define!(T) = comptime!(match depth {
            0 => ElemType::Float(FloatKind::F16),
            _ => ElemType::Float(FloatKind::F32),
        });
        if depth > 0 {
            named_after_reentry(depth - 1);
        }
        elem_type_of::<T>()
    }

    /// A function re-entered on another element finds its own definition again when the inner
    /// call returns: an op on one tile that runs itself on a tile of another element must not
    /// go on at the inner element.
    #[test]
    fn a_definition_lasts_to_the_end_of_its_block() {
        let scope = Scope::root(KernelSettings::new(
            Dim3::new_single(),
            ExecutionMode::Checked,
            AddressType::U32,
        ));
        let named = named_after_reentry::expand(&scope, 1);
        assert_eq!(named, ElemType::Float(FloatKind::F32));
    }

    /// What `N` sizes after the call one depth down has sized it again: two at the bottom, four
    /// above it.
    #[cube]
    fn sized_after_reentry(#[comptime] depth: u32) -> comptime_type!(usize) {
        let size!(N) = comptime!(match depth {
            0 => 2usize,
            _ => 4usize,
        });
        if depth > 0 {
            sized_after_reentry(depth - 1);
        }
        intrinsic!(|scope| N::__expand_value(scope))
    }

    /// A size's definition lasts to the end of its block as a type's does: a function re-entered
    /// on another size finds its own again when the inner call returns.
    #[test]
    fn a_size_lasts_to_the_end_of_its_block() {
        let scope = Scope::root(KernelSettings::new(
            Dim3::new_single(),
            ExecutionMode::Checked,
            AddressType::U32,
        ));
        assert_eq!(sized_after_reentry::expand(&scope, 1), 4);
    }
}
