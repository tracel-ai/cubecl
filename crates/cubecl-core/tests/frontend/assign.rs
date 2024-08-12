use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn mut_assign() {
    let mut x = UInt::new(0);
    x += UInt::new(1);
}

#[cube]
pub fn mut_assign_input(y: UInt) -> UInt {
    let mut x = y;
    x += UInt::new(1);
    y + UInt::new(2)
}

#[cube]
pub fn assign_mut_input(mut y: UInt) -> UInt {
    let x = y;
    y += UInt::new(1);
    x + UInt::new(2)
}

#[cube]
pub fn assign_vectorized(y: UInt) -> UInt {
    let vectorization_factor = Comptime::vectorization(&y);
    let x = UInt::vectorized(1, Comptime::get(vectorization_factor));
    x + y
}

#[cube]
pub fn assign_deref(y: &mut UInt) -> UInt {
    *y = UInt::new(1);
    *y
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Elem, Item, Operation, Variable},
    };

    #[test]
    fn cube_mut_assign_test() {
        let mut context = CubeContext::root();

        mut_assign::__expand(&mut context);
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_mut_assign());
    }

    #[test]
    fn cube_mut_assign_input_test() {
        let mut context = CubeContext::root();

        let y = context.create_local(Item::new(UInt::as_elem()));

        mut_assign_input::__expand(&mut context, y.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_mut_assign_input());
    }

    #[test]
    fn cube_assign_mut_input_test() {
        let mut context = CubeContext::root();

        let y = context.create_local(Item::new(UInt::as_elem()));

        assign_mut_input::__expand(&mut context, y.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_assign_mut_input());
    }

    #[test]
    fn cube_assign_vectorized_test() {
        let mut context = CubeContext::root();

        let y = context.create_local(Item::vectorized(UInt::as_elem(), 4));

        assign_vectorized::__expand(&mut context, y.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_assign_vectorized());
    }

    #[test]
    fn cube_assign_deref_test() {
        let mut context = CubeContext::root();

        let y = context.create_local(Item::new(UInt::as_elem()));
        assign_deref::__expand(&mut context, y.into());

        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_assign_deref());
    }

    fn inline_macro_ref_mut_assign() -> Vec<Operation> {
        let context = CubeContext::root();

        let mut scope = context.into_scope();
        let x = scope.create_local(Item::new(Elem::UInt));

        let zero: Variable = 0u32.into();
        let one: Variable = 1u32.into();

        cpa!(scope, x = zero);
        cpa!(scope, x = x + one);

        scope.operations
    }

    fn inline_macro_ref_mut_assign_input() -> Vec<Operation> {
        let mut context = CubeContext::root();
        let item = Item::new(Elem::UInt);
        let y = context.create_local(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let one: Variable = 1u32.into();
        let two: Variable = 2u32.into();

        cpa!(scope, x = y);
        cpa!(scope, x = x + one);
        cpa!(scope, y = y + two);

        scope.operations
    }

    fn inline_macro_ref_assign_mut_input() -> Vec<Operation> {
        let mut context = CubeContext::root();
        let item = Item::new(Elem::UInt);
        let y = context.create_local(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let one: Variable = 1u32.into();
        let two: Variable = 2u32.into();

        cpa!(scope, x = y);
        cpa!(scope, y = y + one);
        cpa!(scope, x = x + two);

        scope.operations
    }

    fn inline_macro_ref_assign_vectorized() -> Vec<Operation> {
        let mut context = CubeContext::root();
        let item = Item::vectorized(Elem::UInt, 4);
        let y = context.create_local(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let zero: Variable = 0u32.into();
        let one: Variable = 1u32.into();
        let two: Variable = 2u32.into();
        let three: Variable = 3u32.into();

        cpa!(scope, x[zero] = one);
        cpa!(scope, x[one] = one);
        cpa!(scope, x[two] = one);
        cpa!(scope, x[three] = one);
        cpa!(scope, x = x + y);

        scope.operations
    }

    fn inline_macro_ref_assign_deref() -> Vec<Operation> {
        let context = CubeContext::root();
        let mut scope = context.into_scope();
        let y = scope.create_local(Item::new(Elem::UInt));

        let one: Variable = 1u32.into();

        cpa!(scope, y = one);

        scope.operations
    }
}
