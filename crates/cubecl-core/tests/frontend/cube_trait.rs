use cubecl_core as cubecl;
use cubecl_core::prelude::*;

// #[cube(debug)]
// trait FunctionGeneriAc {
//     #[allow(unused)]
//     fn test<C: Float>(&self, lhs: C, rhs: C) -> C;
// }
//

trait TestF {
    fn test<C: Float>(&self, lhs: C, rhs: C) -> C;

    fn __expand_test<C: Float>(
        context: &mut cubecl::prelude::CubeContext,
        v: &Self::ExpandType,
        lhs: <C as cubecl::prelude::CubeType>::ExpandType,
        rhs: <C as cubecl::prelude::CubeType>::ExpandType,
    ) -> <C as cubecl::prelude::CubeType>::ExpandType
    where
        Self: CubeType,
        Self::ExpandType: TestF,
    {
        v.__expand_test_method::<C>(context, lhs, rhs)
    }

    fn __expand_test_method<C: Float>(
        &self,
        context: &mut cubecl::prelude::CubeContext,
        lhs: <C as cubecl::prelude::CubeType>::ExpandType,
        rhs: <C as cubecl::prelude::CubeType>::ExpandType,
    ) -> <C as cubecl::prelude::CubeType>::ExpandType;
}

#[derive(CubeType)]
struct TestImpl;

impl TestF for TestImpl {
    fn test<C: Float>(&self, lhs: C, rhs: C) -> C {
        lhs + rhs
    }

    fn __expand_test_method<C: Float>(
        &self,
        context: &mut cubecl_core::prelude::CubeContext,
        lhs: <C as cubecl_core::prelude::CubeType>::ExpandType,
        rhs: <C as cubecl_core::prelude::CubeType>::ExpandType,
    ) -> <C as cubecl_core::prelude::CubeType>::ExpandType {
        panic!("Unsupported");
    }
}

impl TestF for <TestImpl as CubeType>::ExpandType {
    fn test<C: Float>(&self, lhs: C, rhs: C) -> C {
        panic!("Unsupported");
    }

    fn __expand_test_method<C: Float>(
        &self,
        context: &mut cubecl_core::prelude::CubeContext,
        lhs: <C as cubecl_core::prelude::CubeType>::ExpandType,
        rhs: <C as cubecl_core::prelude::CubeType>::ExpandType,
    ) -> <C as cubecl_core::prelude::CubeType>::ExpandType {
        // Would put the generated code.
        todo!();
    }
}

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
impl FunctionGeneric for TestF {
    fn test<C: Float>(lhs: C, rhs: C) -> C {
        lhs + rhs
    }
}

#[cube]
impl<C: Float> TraitGeneric<C> for TestF {
    fn test(lhs: C, rhs: C) -> C {
        lhs + rhs
    }
}

#[cube]
impl<C: Float> CombinedTraitFunctionGeneric<C> for TestF {
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

        <TestF as FunctionGeneric>::__expand_test::<f32>(&mut context, lhs.into(), rhs.into());

        assert_eq!(simple_scope(), context.into_scope());
    }

    #[test]
    fn test_trait_generic() {
        let mut context = CubeContext::default();
        let lhs = context.create_local_binding(Item::new(f32::as_elem()));
        let rhs = context.create_local_binding(Item::new(f32::as_elem()));

        <TestF as TraitGeneric<f32>>::__expand_test(&mut context, lhs.into(), rhs.into());

        assert_eq!(simple_scope(), context.into_scope());
    }

    #[test]
    fn test_combined_function_generic() {
        let mut context = CubeContext::default();
        let lhs = context.create_local_binding(Item::new(f32::as_elem()));
        let rhs = context.create_local_binding(Item::new(f32::as_elem()));

        <TestF as CombinedTraitFunctionGeneric<f32>>::__expand_test::<u32>(
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
