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

#[macro_export]
macro_rules! size {
    ($name: ident) => {
        $name
    };
}

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
