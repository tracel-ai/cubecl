use cubecl_core as cubecl;
use cubecl_core::{
    cube,
    frontend::{Cast, Float, Int, Numeric},
};

#[cube]
pub fn cast_float_kind<F1: Float, F2: Float>(input: F1) {
    let x = input + F1::new(5.9);
    let y = F2::cast_from(x);
    let _ = y + F2::new(2.3);
}

#[cube]
pub fn cast_int_kind<I1: Int, I2: Int>(input: I1) {
    let x = input + I1::new(5);
    let y = I2::cast_from(x);
    let _ = y + I2::new(2);
}

#[cube]
pub fn cast_numeric_to_kind<T: Numeric, I: Int>(input: T) {
    let x = input + T::from_int(5);
    let y = I::cast_from(x);
    let _ = y + I::from_int(2);
}

#[cube]
pub fn cast_int_to_numeric<I: Int, T: Numeric>(input: I) {
    let x = input + I::from_int(5);
    let y = T::cast_from(x);
    let _ = y + T::from_int(2);
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        frontend::{CubeContext, CubePrimitive},
        ir::{Item, Variable},
    };

    #[test]
    fn cube_cast_float_kind_test() {
        let mut context = CubeContext::default();
        let item = Item::new(f64::as_elem());

        let input = context.create_local_binding(item);

        cast_float_kind::expand::<f64, f32>(&mut context, input.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_float());
    }

    #[test]
    fn cube_cast_int_kind_test() {
        let mut context = CubeContext::default();
        let item = Item::new(i32::as_elem());

        let input = context.create_local_binding(item);

        cast_int_kind::expand::<i32, i64>(&mut context, input.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    #[test]
    fn cube_cast_numeric_kind_test() {
        let mut context = CubeContext::default();
        let item = Item::new(i32::as_elem());

        let input = context.create_local_binding(item);

        cast_numeric_to_kind::expand::<i32, i64>(&mut context, input.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    #[test]
    fn cube_cast_kind_numeric_test() {
        let mut context = CubeContext::default();
        let item = Item::new(i32::as_elem());

        let input = context.create_local_binding(item);

        cast_int_to_numeric::expand::<i32, i64>(&mut context, input.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref_int());
    }

    fn inline_macro_ref_float() -> String {
        let mut context = CubeContext::default();
        let float_64 = Item::new(f64::as_elem());
        let float_32 = Item::new(f32::as_elem());
        let input = context.create_local_binding(float_64);

        let mut scope = context.into_scope();
        let input: Variable = input.into();
        let y = scope.create_local(float_32);

        cpa!(scope, input = input + 5.9f32 as f64);
        cpa!(scope, y = cast(input));
        cpa!(scope, y = y + 2.3f32);

        format!("{:?}", scope.operations)
    }

    fn inline_macro_ref_int() -> String {
        let mut context = CubeContext::default();
        let int_32 = Item::new(i32::as_elem());
        let int_64 = Item::new(i64::as_elem());
        let input = context.create_local_binding(int_32);

        let mut scope = context.into_scope();
        let input: Variable = input.into();
        let y = scope.create_local(int_64);

        cpa!(scope, input = input + 5i32);
        cpa!(scope, y = cast(input));
        cpa!(scope, y = y + 2i64);

        format!("{:?}", scope.operations)
    }
}
