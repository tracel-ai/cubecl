use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn redeclare_same_scope<I: Int>(mut x: I) {
    let i = I::new(1);
    x += i;
    let i = I::new(2);
    x += i;
}

#[cube]
pub fn redeclare_same_scope_other_type<I: Int, F: Float>(mut x: I) -> F {
    let i = I::new(1);
    x += i;
    let i = F::new(2.);
    i + i
}

#[cube]
pub fn redeclare_different_scope<I: Int>(mut x: I) {
    let y = I::new(1);
    x += y;
    for _ in 0..2u32 {
        let y = I::new(2);
        x += y;
    }
}

#[cube]
#[allow(unused)]
pub fn redeclare_two_for_loops(mut x: u32) {
    for i in 0..2 {
        x += i;
    }
    for i in 0..2 {
        x += i;
        x += i;
    }
}

mod tests {
    use cubecl_core::{
        cpa,
        ir::{Item, Variable},
    };
    use pretty_assertions::assert_eq;

    use super::*;

    type ElemType = i32;

    #[test]
    fn cube_redeclare_same_scope_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));

        redeclare_same_scope::expand::<ElemType>(&mut context, x.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_same_scope()
        );
    }

    #[test]
    fn cube_redeclare_same_scope_other_type_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));

        redeclare_same_scope_other_type::expand::<ElemType, f32>(&mut context, x.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_same_scope_other_type()
        );
    }

    #[test]
    fn cube_redeclare_different_scope_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));

        redeclare_different_scope::expand::<ElemType>(&mut context, x.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_different()
        );
    }

    #[test]
    fn cube_redeclare_two_for_loops_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(u32::as_elem()));

        redeclare_two_for_loops::expand(&mut context, x.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            inline_macro_ref_two_for_loops()
        );
    }

    fn inline_macro_ref_same_scope() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let x = context.create_local_binding(item);
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        let value: ExpandElement = ElemType::from(1).into();
        let value: Variable = *value;

        cpa!(scope, x += value);

        let value: ExpandElement = ElemType::from(2).into();
        let value: Variable = *value;

        cpa!(scope, x += value);

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_same_scope_other_type() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let x = context.create_local_binding(item);
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        let i: ExpandElement = ElemType::new(1).into();
        let i = *i;
        cpa!(scope, x += i);
        let i: ExpandElement = 2f32.into();
        let i = *i;
        let y = scope.create_local(Item::new(f32::as_elem()));
        cpa!(scope, y = i + i);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_different() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let x = context.create_local_binding(item);
        let end = 2u32;
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        let y: ExpandElement = ElemType::new(1).into();
        let y = *y;
        cpa!(scope, x += y);

        cpa!(
            &mut scope,
            range(0u32, end, false).for_each(|_, scope| {
                let value: ExpandElement = ElemType::new(2).into();
                let value: Variable = *value;

                cpa!(scope, x += value);
            })
        );

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_two_for_loops() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(u32::as_elem());

        let x = context.create_local_binding(item);
        let end = 2u32;
        let mut scope = context.into_scope();
        let x: Variable = x.into();

        cpa!(
            &mut scope,
            range(0u32, end, false).for_each(|i, scope| {
                cpa!(scope, x += i);
            })
        );

        cpa!(
            &mut scope,
            range(0u32, end, false).for_each(|i, scope| {
                cpa!(scope, x += i);
                cpa!(scope, x += i);
            })
        );

        format!("{:?}", scope.operations)
    }
}
