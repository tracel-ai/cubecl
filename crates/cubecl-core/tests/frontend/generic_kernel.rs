use cubecl_core as cubecl;
use cubecl_core::{cube, frontend::Numeric};

#[cube]
pub fn generic_kernel<T: Numeric>(lhs: T) {
    let _ = lhs + T::from_int(5);
}

mod tests {
    use cubecl_core::{
        cpa,
        frontend::{CubeContext, CubePrimitive},
        ir::{Item, Variable},
    };

    use super::*;

    #[test]
    fn cube_generic_float_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(f32::as_elem()));

        generic_kernel::expand::<f32>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
    }

    #[test]
    fn cube_generic_int_test() {
        let mut context = CubeContext::default();

        let lhs = context.create_local_binding(Item::new(i32::as_elem()));

        generic_kernel::expand::<i32>(&mut context, lhs.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    fn inline_macro_ref_float() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(f32::as_elem());
        let var = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let var: Variable = var.into();
        cpa!(scope, var = var + 5.0f32);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_int() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(i32::as_elem());
        let var = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let var: Variable = var.into();
        cpa!(scope, var = var + 5);

        format!("{:?}", scope.operations)
    }
}
