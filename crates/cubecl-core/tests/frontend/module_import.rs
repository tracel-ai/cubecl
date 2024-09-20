use cubecl_core as cubecl;
use cubecl_core::prelude::*;

mod elsewhere {
    use super::*;

    #[cube]
    pub fn my_func<F: Float>(x: F) -> F {
        x * F::from_int(2)
    }
}

mod here {
    use super::*;

    #[cube]
    pub fn caller<F: Float>(x: F) {
        let _ = x + elsewhere::my_func::<F>(x);
    }

    #[cube]
    pub fn no_call_ref<F: Float>(x: F) {
        let _ = x + x * F::from_int(2);
    }
}

mod tests {
    use super::*;
    use cubecl_core::ir::Item;

    type ElemType = f32;

    #[test]
    fn cube_call_equivalent_to_no_call_no_arg_test() {
        let mut caller_context = CubeContext::default();
        let x = caller_context.create_local_binding(Item::new(ElemType::as_elem()));
        here::caller::expand::<ElemType>(&mut caller_context, x.into());
        let caller_scope = caller_context.into_scope();

        let mut no_call_context = CubeContext::default();
        let x = no_call_context.create_local_binding(Item::new(ElemType::as_elem()));
        here::no_call_ref::expand::<ElemType>(&mut no_call_context, x.into());
        let no_call_scope = no_call_context.into_scope();

        assert_eq!(
            format!("{:?}", caller_scope.operations),
            format!("{:?}", no_call_scope.operations)
        );
    }
}
