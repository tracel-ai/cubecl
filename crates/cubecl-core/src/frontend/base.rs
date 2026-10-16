use cubecl_ir::Scope;

use crate::frontend::{CubePrimitive, NativeExpand, init_mut_of_type};

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
pub fn uninit_local<T: CubePrimitive>(scope: &Scope) -> NativeExpand<T> {
    init_mut_of_type(scope, T::__expand_as_type(scope)).into()
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
        if comptime!(depth > 0) {
            named_after_reentry(comptime!(depth - 1));
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
        if comptime!(depth > 0) {
            sized_after_reentry(comptime!(depth - 1));
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
