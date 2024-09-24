use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn while_not<I: Int>(lhs: I) {
    while lhs != I::from_int(0) {
        let _ = lhs % I::from_int(1);
    }
}

#[cube]
pub fn manual_loop_break<I: Int>(lhs: I) {
    loop {
        if lhs == I::from_int(0) {
            break;
        }
        let _ = lhs % I::from_int(1);
    }
}

#[cube]
pub fn loop_with_return<I: Int>(lhs: I) {
    loop {
        if lhs == I::from_int(0) {
            return;
        }
        let _ = lhs % I::from_int(1);
    }
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Branch, Elem, Item, Variable},
    };
    use pretty_assertions::assert_eq;

    type ElemType = i32;

    #[test]
    fn cube_while_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        while_not::expand::<ElemType>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:#?}", scope.operations), inline_macro_ref_while());
    }

    #[test]
    fn cube_loop_break_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        manual_loop_break::expand::<ElemType>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_loop(false)
        );
    }

    #[test]
    fn cube_loop_with_return_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        loop_with_return::expand::<ElemType>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_loop(true)
        );
    }

    fn inline_macro_ref_while() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let y = scope.create_local(item);
        let lhs: Variable = lhs.into();

        cpa!(
            &mut scope,
            loop(|scope| {
                cpa!(scope, cond = lhs != 0);
                cpa!(scope, cond = !cond);
                cpa!(scope, if(cond).then(|scope|{
                        scope.register(Branch::Break)
                }));
                // Must not mutate `lhs` because it is used in every iteration
                cpa!(scope, y = lhs % 1i32);
            })
        );

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_loop(is_return: bool) -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let y = scope.create_local(item);
        let lhs: Variable = lhs.into();

        cpa!(
            &mut scope,
            loop(|scope| {
                cpa!(scope, cond = lhs == 0);
                cpa!(scope, if(cond).then(|scope|{
                    match is_return {
                        true => scope.register(Branch::Return),
                        false => scope.register(Branch::Break)
                    }
                }));
                // Must not mutate `lhs` because it is used in every iteration
                cpa!(scope, y = lhs % 1i32);
            })
        );

        format!("{:?}", scope.operations)
    }
}
