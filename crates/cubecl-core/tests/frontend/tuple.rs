use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn tuple_const() -> (u32, u32) {
    let x = 0u32;
    let y = 1u32;
    (x, y)
}

#[cube]
pub fn tuple_destructuring() -> (u32, u32) {
    let x = (0u32, 1u32);
    let (a, b) = x;
    (a + 1, b)
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Elem, Item, Operation, Variable},
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn cube_tuple_const_test() {
        let mut context = CubeContext::default();

        tuple_const::expand(&mut context);
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_tuple_const());
    }

    fn inline_macro_ref_tuple_const() -> Vec<Operation> {
        let context = CubeContext::default();

        let mut scope = context.into_scope();
        let x = scope.create_local(Item::new(Elem::UInt));
        let y = scope.create_local(Item::new(Elem::UInt));

        let zero: Variable = 0u32.into();
        let one: Variable = 1u32.into();

        cpa!(scope, x = zero);
        cpa!(scope, y = one);

        scope.operations
    }

    #[test]
    fn cube_tuple_destructuring() {
        let mut context = CubeContext::default();

        tuple_destructuring::expand(&mut context);
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_tuple_destructuring());
    }

    fn inline_macro_ref_tuple_destructuring() -> Vec<Operation> {
        let context = CubeContext::default();

        let mut scope = context.into_scope();
        let a = scope.create_local(Item::new(Elem::UInt));
        let b = scope.create_local(Item::new(Elem::UInt));

        let one: Variable = 1u32.into();

        cpa!(scope, a = one);
        cpa!(scope, b = one);

        scope.operations
    }
}
