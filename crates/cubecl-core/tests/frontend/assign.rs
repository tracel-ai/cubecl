#![allow(unused)]

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn mut_assign() {
    let mut x: u32 = 0;
    x += 1;
}

#[cube]
pub fn mut_assign_input(y: u32) -> u32 {
    let mut x = y;
    x += 1;
    y + 2
}

#[cube]
pub fn assign_mut_input(mut y: u32) -> u32 {
    let x = y;
    y += 1;
    x + 2
}

#[cube]
pub fn assign_vectorized(y: u32) -> u32 {
    let x = u32::vectorized(1, vectorization_of(&y));
    x + y
}

#[cube]
pub fn assign_deref(y: &mut u32) -> u32 {
    *y = 1;
    *y
}

mod tests {
    use pretty_assertions::assert_eq;
    use std::num::NonZero;

    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Elem, Item, Operation, Variable},
    };

    #[test]
    fn cube_mut_assign_test() {
        let mut context = CubeContext::default();

        mut_assign::expand(&mut context);
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_mut_assign());
    }

    #[test]
    fn cube_mut_assign_input_test() {
        let mut context = CubeContext::default();

        let y = context.create_local_binding(Item::new(u32::as_elem()));

        mut_assign_input::expand(&mut context, y.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_mut_assign_input());
    }

    #[test]
    fn cube_assign_mut_input_test() {
        let mut context = CubeContext::default();

        let y = context.create_local_binding(Item::new(u32::as_elem()));

        assign_mut_input::expand(&mut context, y.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_assign_mut_input());
    }

    #[test]
    fn cube_assign_vectorized_test() {
        let mut context = CubeContext::default();

        let y = context.create_local_binding(Item::vectorized(u32::as_elem(), NonZero::new(4)));

        assign_vectorized::expand(&mut context, y.into());
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_assign_vectorized());
    }

    #[test]
    fn cube_assign_deref_test() {
        let mut context = CubeContext::default();

        let y = context.create_local_binding(Item::new(u32::as_elem()));
        assign_deref::expand(&mut context, y.into());

        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_assign_deref());
    }

    fn inline_macro_ref_mut_assign() -> Vec<Operation> {
        let context = CubeContext::default();

        let mut scope = context.into_scope();
        let x = scope.create_local(Item::new(Elem::UInt));

        let zero: Variable = 0u32.into();
        let one: Variable = 1u32.into();

        cpa!(scope, x = zero);
        cpa!(scope, x = x + one);

        scope.operations
    }

    fn inline_macro_ref_mut_assign_input() -> Vec<Operation> {
        let mut context = CubeContext::default();
        let item = Item::new(Elem::UInt);
        let y = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let one: Variable = 1u32.into();
        let two: Variable = 2u32.into();

        cpa!(scope, x = y);
        cpa!(scope, x = x + one);
        cpa!(scope, x = y + two);

        scope.operations
    }

    fn inline_macro_ref_assign_mut_input() -> Vec<Operation> {
        let mut context = CubeContext::default();
        let item = Item::new(Elem::UInt);
        let y = context.create_local_variable(item);
        println!("{:?}", y.index());

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
        let mut context = CubeContext::default();
        let item = Item::vectorized(Elem::UInt, NonZero::new(4));
        let y = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let y: Variable = y.into();
        let x = scope.create_local(item);

        let zero: Variable = 0u32.into();
        let one: Variable = 1u32.into();
        let two: Variable = 2u32.into();
        let three: Variable = 3u32.into();

        cpa!(scope, x = one);
        cpa!(scope, x = x + y);

        scope.operations
    }

    fn inline_macro_ref_assign_deref() -> Vec<Operation> {
        let context = CubeContext::default();
        let mut scope = context.into_scope();
        let y = scope.create_local(Item::new(Elem::UInt));

        let one: Variable = 1u32.into();

        cpa!(scope, y = one);

        scope.operations
    }
}
