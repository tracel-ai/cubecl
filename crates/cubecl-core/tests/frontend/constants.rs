use half::{bf16, f16};
use paste::paste;

use cubecl_core::{self as cubecl, prelude::*};

macro_rules! gen_cube {
    ($trait:ident, [ $($constant:ident $(| $ret_type:ty)?),* ]) => {
        $(
            gen_cube!($trait, $constant, $($ret_type)?);
        )*
    };
    ($trait:ident, $constant:ident,) => {
        gen_cube!($trait, $constant, T);
    };
    ($trait:ident, $constant:ident, $ret_type:ty) => {
        paste! {
            gen_cube!([< $trait:lower _ $constant:lower >], $trait, $constant, $ret_type);
        }
    };
    ($func_name:ident, $trait:ident, $constant:ident, $ret_type:ty) => {
        #[cube]
        pub fn $func_name<T: $trait>() -> $ret_type {
            T::$constant
        }
    };
}

macro_rules! gen_tests {
    ($trait:ident, [ $($type:ident),* ], $constants:tt) => {
        $(
            gen_tests!($trait, $type, $constants);
        )*
    };
    ($trait:ident, $type:ident, [ $($constant:ident $(| $ret_type:ty)?),* ]) => {
        $(
            gen_tests!($trait, $type, $constant, $($ret_type)?);
        )*
    };
    ($trait:ident, $type:ident, $constant:ident,) => {
        gen_tests!($trait, $type, $constant, $type);
    };
    ($trait:ident, $type:ident, $constant:ident, $ret_type:ty) => {
        paste! {
            gen_tests!([< cube_ $trait:lower _ $constant:lower _ $type _test >], [< $trait:lower _ $constant:lower >], $type, $constant, $ret_type);
        }
    };
    ($test_name:ident, $func_name:ident, $type:ty, $constant:ident, $ret_type:ty) => {
        #[test]
        fn $test_name() {
            let mut context = CubeContext::default();
            $func_name::expand::<$type>(&mut context);
            let scope = context.into_scope();

            let mut scope1 = CubeContext::default().into_scope();
            let item = Item::new(<$ret_type>::as_elem());
            scope1.create_with_value(<$type>::$constant, item);

            assert_eq!(
                format!("{:?}", scope.operations),
                format!("{:?}", scope1.operations)
            );
        }
    };
}

gen_cube!(Numeric, [MAX, MIN]);
gen_cube!(Int, [BITS | u32]);
gen_cube!(
    Float,
    [
        DIGITS | u32,
        EPSILON,
        INFINITY,
        MANTISSA_DIGITS | u32,
        MAX_10_EXP | i32,
        MAX_EXP | i32,
        MIN_10_EXP | i32,
        MIN_EXP | i32,
        MIN_POSITIVE,
        NAN,
        NEG_INFINITY,
        RADIX | u32
    ]
);

mod tests {
    use super::*;
    use cubecl_core::{
        frontend::{CubeContext, CubePrimitive},
        ir::Item,
    };
    use pretty_assertions::assert_eq;

    gen_tests!(Numeric, [bf16, f16, f32, f64, i32, i64, u32], [MAX, MIN]);
    gen_tests!(Int, [i32, i64, u32], [BITS | u32]);
    gen_tests!(
        Float,
        [bf16, f16, f32, f64],
        [
            DIGITS | u32,
            EPSILON,
            INFINITY,
            MANTISSA_DIGITS | u32,
            MAX_10_EXP | i32,
            MAX_EXP | i32,
            MIN_10_EXP | i32,
            MIN_EXP | i32,
            MIN_POSITIVE,
            NAN,
            NEG_INFINITY,
            RADIX | u32
        ]
    );
}
