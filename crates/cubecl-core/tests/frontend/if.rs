use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn if_greater<T: Numeric>(lhs: T) {
    if lhs > T::from_int(0) {
        let _ = lhs + T::from_int(4);
    }
}

#[cube]
pub fn if_greater_var<T: Numeric>(lhs: T) {
    let x = lhs > T::from_int(0);
    if x {
        let _ = lhs + T::from_int(4);
    }
}

#[cube]
pub fn if_then_else<F: Float>(lhs: F) {
    if lhs < F::from_int(0) {
        let _ = lhs + F::from_int(4);
    } else {
        let _ = lhs - F::from_int(5);
    }
}

#[cube]
pub fn elsif<F: Float>(lhs: F) {
    if lhs < F::new(0.) {
        let _ = lhs + F::new(2.);
    } else if lhs > F::new(0.) {
        let _ = lhs + F::new(1.);
    } else {
        let _ = lhs + F::new(0.);
    }
}

#[cube]
pub fn elsif_assign<F: Float>(lhs: F) {
    let _ = if lhs < F::new(0.) {
        lhs + F::new(2.)
    } else if lhs > F::new(0.) {
        lhs + F::new(1.)
    } else {
        lhs + F::new(0.)
    };
}

mod tests {
    use cubecl_core::{
        cpa,
        frontend::{CubeContext, CubePrimitive},
        ir::{Elem, Item, Variable},
    };
    use pretty_assertions::assert_eq;

    use super::*;

    type ElemType = f32;

    #[test]
    fn cube_if_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        if_greater::expand::<ElemType>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:#?}", scope.operations), inline_macro_ref_if());
    }

    #[test]
    fn cube_if_else_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        if_then_else::expand::<ElemType>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_if_else()
        );
    }

    #[test]
    fn cube_elsif_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        elsif::expand::<ElemType>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:#?}", scope.operations), inline_macro_ref_elsif());
    }

    #[test]
    fn cube_elsif_assign_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(ElemType::as_elem()));

        elsif_assign::expand::<ElemType>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_elsif_assign()
        );
    }

    fn inline_macro_ref_if() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let lhs: Variable = lhs.into();

        cpa!(scope, cond = lhs > 0f32);
        cpa!(&mut scope, if(cond).then(|scope| {
            cpa!(scope, lhs = lhs + 4.0f32);
        }));

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_if_else() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let cond = scope.create_local(Item::new(Elem::Bool));
        let lhs: Variable = lhs.into();
        let y = scope.create_local(item);

        cpa!(scope, cond = lhs < 0f32);
        cpa!(&mut scope, if(cond).then(|scope| {
            cpa!(scope, y = lhs + 4.0f32);
        }).else(|scope|{
            cpa!(scope, y = lhs - 5.0f32);
        }));

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_elsif() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let cond1 = scope.create_local(Item::new(Elem::Bool));
        let lhs: Variable = lhs.into();
        let y = scope.create_local(item);
        let cond2 = scope.create_local(Item::new(Elem::Bool));

        cpa!(scope, cond1 = lhs < 0f32);
        cpa!(&mut scope, if(cond1).then(|scope| {
            cpa!(scope, y = lhs + 2.0f32);
        }).else(|mut scope|{
            cpa!(scope, cond2 = lhs > 0f32);
            cpa!(&mut scope, if(cond2).then(|scope| {
                cpa!(scope, y = lhs + 1.0f32);
            }).else(|scope|{
                cpa!(scope, y = lhs + 0.0f32);
            }));
        }));

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_elsif_assign() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let lhs = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let lhs: Variable = lhs.into();
        let cond1 = scope.create_local(Item::new(Elem::Bool));
        let y = scope.create_local(item);
        let out = scope.create_local(item);
        let cond2 = scope.create_local(Item::new(Elem::Bool));
        let out2 = scope.create_local(item);

        cpa!(scope, cond1 = lhs < 0f32);
        cpa!(&mut scope, if(cond1).then(|scope| {
            cpa!(scope, y = lhs + 2.0f32);
            cpa!(scope, out = y);
        }).else(|mut scope|{
            cpa!(scope, cond2 = lhs > 0f32);
            cpa!(&mut scope, if(cond2).then(|scope| {
                cpa!(scope, y = lhs + 1.0f32);
                cpa!(scope, out2 = y);
            }).else(|scope|{
                cpa!(scope, y = lhs + 0.0f32);
                cpa!(scope, out2 = y);
            }));
            cpa!(scope, out = out2);
        }));

        format!("{:#?}", scope.operations)
    }
}
