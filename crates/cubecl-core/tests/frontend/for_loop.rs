use cubecl_core as cubecl;
use cubecl_core::{
    cube,
    frontend::{Array, CubeContext, CubePrimitive, Float},
};

type ElemType = f32;

#[cube]
pub fn for_loop<F: Float>(mut lhs: Array<F>, rhs: F, end: u32, #[comptime] unroll: bool) {
    let tmp1 = rhs * rhs;
    let tmp2 = tmp1 + rhs;

    #[unroll(unroll)]
    for i in 0..end {
        lhs[i] = tmp2 + lhs[i];
    }
}

#[cube]
pub fn for_in_loop<F: Float>(input: &Array<F>) -> F {
    let mut sum = F::new(0.0);

    for item in input {
        sum += item;
    }
    sum
}

mod tests {
    use cubecl::frontend::ExpandElement;
    use cubecl_core::{
        cpa,
        ir::{Item, Variable},
    };
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_for_loop_with_unroll() {
        let mut context = CubeContext::default();
        let unroll = true;

        let lhs = context.create_local_array(Item::new(ElemType::as_elem()), 4u32);
        let rhs = context.create_local_binding(Item::new(ElemType::as_elem()));
        let end: ExpandElement = 4u32.into();

        for_loop::expand::<ElemType>(&mut context, lhs.into(), rhs.into(), end.into(), unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:#?}", scope.operations), inline_macro_ref(unroll));
    }

    #[test]
    fn test_for_loop_no_unroll() {
        let mut context = CubeContext::default();
        let unroll = false;

        let lhs = context.create_local_array(Item::new(ElemType::as_elem()), 4u32);
        let rhs = context.create_local_binding(Item::new(ElemType::as_elem()));
        let end: ExpandElement = 4u32.into();

        for_loop::expand::<ElemType>(&mut context, lhs.into(), rhs.into(), end.into(), unroll);
        let scope = context.into_scope();

        assert_eq!(format!("{:#?}", scope.operations), inline_macro_ref(unroll));
    }

    #[test]
    fn test_for_in_loop() {
        let mut context = CubeContext::default();

        let input = context.create_local_array(Item::new(ElemType::as_elem()), 4u32);

        for_in_loop::expand::<ElemType>(&mut context, input.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:#?}", scope.operations),
            inline_macro_ref_for_in()
        );
    }

    fn inline_macro_ref(unroll: bool) -> String {
        let context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let lhs = scope.create_local_array(item, 4u32);
        let rhs = scope.create_local(item);
        let end = 4u32;

        // Kernel
        let tmp1 = scope.create_local(item);
        cpa!(scope, tmp1 = rhs * rhs);
        cpa!(scope, tmp1 = tmp1 + rhs);

        cpa!(
            &mut scope,
            range(0u32, end, unroll).for_each(|i, scope| {
                cpa!(scope, rhs = lhs[i]);
                cpa!(scope, rhs = tmp1 + rhs);
                cpa!(scope, lhs[i] = rhs);
            })
        );

        format!("{:#?}", scope.operations)
    }

    fn inline_macro_ref_for_in() -> String {
        let context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let input = scope.create_local_array(item, 4u32);
        let sum = scope.create_local(item);
        let end = scope.create_local(Item::new(u32::as_elem()));
        let zero: Variable = ElemType::new(0.0).into();

        // Kernel
        let tmp1 = scope.create_local(item);
        cpa!(scope, sum = zero);
        cpa!(scope, end = len(input));

        cpa!(
            &mut scope,
            range(0u32, end).for_each(|i, scope| {
                cpa!(scope, tmp1 = input[i]);
                cpa!(scope, sum = sum + tmp1);
            })
        );

        format!("{:#?}", scope.operations)
    }
}
