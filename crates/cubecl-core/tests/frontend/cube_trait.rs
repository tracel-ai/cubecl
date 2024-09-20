use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
trait FunctionGeneric {
    #[allow(unused)]
    fn test<C: Float>(lhs: C, rhs: C) -> C;
}

#[cube]
trait TraitGeneric<C: Float> {
    #[allow(unused)]
    fn test(lhs: C, rhs: C) -> C;
}

#[cube]
trait CombinedTraitFunctionGeneric<C: Float> {
    #[allow(unused)]
    fn test<O: Numeric>(lhs: C, rhs: C) -> O;
}

struct Test;

#[cube]
impl FunctionGeneric for Test {
    fn test<C: Float>(lhs: C, rhs: C) -> C {
        lhs + rhs
    }
}

#[cube]
impl<C: Float> TraitGeneric<C> for Test {
    fn test(lhs: C, rhs: C) -> C {
        lhs + rhs
    }
}

#[cube]
impl<C: Float> CombinedTraitFunctionGeneric<C> for Test {
    fn test<O: Numeric>(lhs: C, rhs: C) -> O {
        O::cast_from(lhs + rhs)
    }
}

#[cube]
pub fn simple<C: Float>(lhs: C, rhs: C) -> C {
    lhs + rhs
}

#[cube]
pub fn with_cast<C: Float, O: Numeric>(lhs: C, rhs: C) -> O {
    O::cast_from(lhs + rhs)
}

mod tests {
    use cubecl_core::ir::{Item, Scope};

    use super::*;

    #[test]
    fn test_function_generic() {
        let mut context = CubeContext::default();
        let lhs = context.create_local_binding(Item::new(f32::as_elem()));
        let rhs = context.create_local_binding(Item::new(f32::as_elem()));

        <Test as FunctionGeneric>::__expand_test::<f32>(&mut context, lhs.into(), rhs.into());

        assert_eq!(simple_scope(), context.into_scope());
    }

    #[test]
    fn test_trait_generic() {
        let mut context = CubeContext::default();
        let lhs = context.create_local_binding(Item::new(f32::as_elem()));
        let rhs = context.create_local_binding(Item::new(f32::as_elem()));

        <Test as TraitGeneric<f32>>::__expand_test(&mut context, lhs.into(), rhs.into());

        assert_eq!(simple_scope(), context.into_scope());
    }

    #[test]
    fn test_combined_function_generic() {
        let mut context = CubeContext::default();
        let lhs = context.create_local_binding(Item::new(f32::as_elem()));
        let rhs = context.create_local_binding(Item::new(f32::as_elem()));

        <Test as CombinedTraitFunctionGeneric<f32>>::__expand_test::<u32>(
            &mut context,
            lhs.into(),
            rhs.into(),
        );

        assert_eq!(with_cast_scope(), context.into_scope());
    }

    fn simple_scope() -> Scope {
        let mut context_ref = CubeContext::default();
        let lhs = context_ref.create_local_binding(Item::new(f32::as_elem()));
        let rhs = context_ref.create_local_binding(Item::new(f32::as_elem()));

        simple::expand::<f32>(&mut context_ref, lhs.into(), rhs.into());
        context_ref.into_scope()
    }

    fn with_cast_scope() -> Scope {
        let mut context_ref = CubeContext::default();
        let lhs = context_ref.create_local_binding(Item::new(f32::as_elem()));
        let rhs = context_ref.create_local_binding(Item::new(f32::as_elem()));

        with_cast::expand::<f32, u32>(&mut context_ref, lhs.into(), rhs.into());
        context_ref.into_scope()
    }
}
