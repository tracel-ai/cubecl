use cubecl_core as cubecl;
use cubecl_core::{cube, frontend::Numeric};

#[cube]
pub fn caller_no_arg(x: u32) {
    let _ = x + callee_no_arg();
}

#[cube]
pub fn callee_no_arg() -> u32 {
    u32::from_int(8)
}

#[cube]
pub fn no_call_no_arg(x: u32) {
    let _ = x + u32::from_int(8);
}

#[cube]
pub fn caller_with_arg(x: u32) {
    let _ = x + callee_with_arg(x);
}

#[cube]
pub fn callee_with_arg(x: u32) -> u32 {
    x * u32::from_int(8)
}

#[cube]
pub fn no_call_with_arg(x: u32) {
    let _ = x + x * u32::from_int(8);
}

#[cube]
pub fn caller_with_generics<T: Numeric>(x: T) {
    let _ = x + callee_with_generics::<T>(x);
}

#[cube]
pub fn callee_with_generics<T: Numeric>(x: T) -> T {
    x * T::from_int(8)
}

#[cube]
pub fn no_call_with_generics<T: Numeric>(x: T) {
    let _ = x + x * T::from_int(8);
}

mod tests {
    use super::*;
    use cubecl_core::{
        frontend::{CubeContext, CubePrimitive},
        ir::{Elem, Item},
    };

    #[test]
    fn cube_call_equivalent_to_no_call_no_arg_test() {
        let mut caller_context = CubeContext::default();
        let x = caller_context.create_local_binding(Item::new(Elem::UInt));
        caller_no_arg::expand(&mut caller_context, x.into());
        let caller_scope = caller_context.into_scope();

        let mut no_call_context = CubeContext::default();
        let x = no_call_context.create_local_binding(Item::new(Elem::UInt));
        no_call_no_arg::expand(&mut no_call_context, x.into());
        let no_call_scope = no_call_context.into_scope();

        assert_eq!(
            format!("{:?}", caller_scope.operations),
            format!("{:?}", no_call_scope.operations)
        );
    }

    #[test]
    fn cube_call_equivalent_to_no_call_with_arg_test() {
        let mut caller_context = CubeContext::default();

        let x = caller_context.create_local_binding(Item::new(Elem::UInt));
        caller_with_arg::expand(&mut caller_context, x.into());
        let caller_scope = caller_context.into_scope();

        let mut no_call_context = CubeContext::default();
        let x = no_call_context.create_local_binding(Item::new(Elem::UInt));
        no_call_with_arg::expand(&mut no_call_context, x.into());
        let no_call_scope = no_call_context.into_scope();

        assert_eq!(
            format!("{:?}", caller_scope.operations),
            format!("{:?}", no_call_scope.operations)
        );
    }

    #[test]
    fn cube_call_equivalent_to_no_call_with_generics_test() {
        let mut caller_context = CubeContext::default();
        type ElemType = i64;
        let x = caller_context.create_local_binding(Item::new(ElemType::as_elem()));
        caller_with_generics::expand::<ElemType>(&mut caller_context, x.into());
        let caller_scope = caller_context.into_scope();

        let mut no_call_context = CubeContext::default();
        let x = no_call_context.create_local_binding(Item::new(ElemType::as_elem()));
        no_call_with_generics::expand::<ElemType>(&mut no_call_context, x.into());
        let no_call_scope = no_call_context.into_scope();

        assert_eq!(
            format!("{:?}", caller_scope.operations),
            format!("{:?}", no_call_scope.operations)
        );
    }
}
