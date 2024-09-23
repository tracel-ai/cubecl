use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeLaunch)]
pub struct EmptyLaunch {}

#[derive(CubeType)]
pub struct EmptyType {}

#[derive(CubeLaunch)]
pub struct UnitLaunch;

#[derive(CubeLaunch)]
pub struct WithField {
    lhs: Array<f32>,
    rhs: Array<f32>,
}

#[derive(CubeLaunch)]
pub struct WithFieldGeneric<F: Float> {
    lhs: Array<F>,
    rhs: Array<F>,
}

#[derive(CubeType)]
pub struct UnitType;

#[derive(CubeType)]
pub struct State<T: Numeric> {
    first: T,
    second: T,
}

#[cube]
pub fn state_receiver_with_reuse<T: Numeric>(state: State<T>) -> T {
    let x = state.first + state.second;
    state.second + x + state.first
}

#[cube]
pub fn attribute_modifier_reuse_field<T: Numeric>(mut state: State<T>) -> T {
    state.first = T::from_int(4);
    state.first
}

#[cube]
pub fn attribute_modifier_reuse_struct<T: Numeric>(mut state: State<T>) -> State<T> {
    state.first = T::from_int(4);
    state
}

#[cube]
fn creator<T: Numeric>(x: T, second: T) -> State<T> {
    let mut state = State::<T> { first: x, second };
    state.second = state.first;

    state
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = f32;

    #[test]
    fn cube_new_struct_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        creator::expand::<ElemType>(&mut context, x.into(), y.into());
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            creator_inline_macro_ref()
        );
    }

    #[test]
    fn cube_struct_as_arg_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        let expanded_state = StateExpand {
            first: x.into(),
            second: y.into(),
        };
        state_receiver_with_reuse::expand::<ElemType>(&mut context, expanded_state);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            receive_state_with_reuse_inline_macro_ref()
        );
    }

    #[test]
    fn cube_struct_assign_to_field_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        let expanded_state = StateExpand {
            first: x.into(),
            second: y.into(),
        };
        attribute_modifier_reuse_field::expand::<ElemType>(&mut context, expanded_state);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            field_modifier_inline_macro_ref()
        );
    }

    #[test]
    fn cube_struct_assign_to_field_reuse_struct_test() {
        let mut context = CubeContext::default();

        let x = context.create_local_binding(Item::new(ElemType::as_elem()));
        let y = context.create_local_binding(Item::new(ElemType::as_elem()));

        let expanded_state = StateExpand {
            first: x.into(),
            second: y.into(),
        };
        attribute_modifier_reuse_struct::expand::<ElemType>(&mut context, expanded_state);
        let scope = context.into_scope();

        assert_eq!(
            format!("{:?}", scope.operations),
            field_modifier_inline_macro_ref()
        );
    }

    fn creator_inline_macro_ref() -> String {
        let context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        let x = scope.create_local(item);
        let y = scope.create_local(item);
        cpa!(scope, y = x);

        format!("{:?}", scope.operations)
    }

    fn field_modifier_inline_macro_ref() -> String {
        let context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());

        let mut scope = context.into_scope();
        scope.create_with_value(4, item);

        format!("{:?}", scope.operations)
    }

    fn receive_state_with_reuse_inline_macro_ref() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local_binding(item);
        let y = context.create_local_binding(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y: Variable = y.into();
        let z = scope.create_local(item);

        cpa!(scope, z = x + y);
        cpa!(scope, z = y + z);
        cpa!(scope, z = z + x);

        format!("{:?}", scope.operations)
    }
}
