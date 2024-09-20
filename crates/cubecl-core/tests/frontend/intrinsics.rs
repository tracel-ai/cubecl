use cubecl_core as cubecl;
use cubecl_core::prelude::*;

fn assert_comptime<T>(_elem: T) {}
pub mod assert_comptime {
    use cubecl_core::prelude::CubeContext;
    pub fn expand<T>(_context: &mut CubeContext, _elem: T) {}
}

#[cube]
pub fn vectorization_of_intrinsic<F: Float>(input: F) -> u32 {
    let vec = vectorization_of(&input);
    assert_comptime::<u32>(vec);
    vec
}

mod tests {
    use pretty_assertions::assert_eq;
    use std::num::NonZero;

    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = f32;

    #[test]
    fn vectorization_of_test() {
        let mut context = CubeContext::default();

        let input =
            context.create_local_binding(Item::vectorized(ElemType::as_elem(), NonZero::new(3)));

        vectorization_of_intrinsic::expand::<ElemType>(&mut context, input.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    fn inline_macro_ref() -> String {
        let mut context = CubeContext::default();
        let item = Item::new(ElemType::as_elem());
        let _input = context.create_local_binding(item);
        let out = context.create_local_binding(Item::new(u32::as_elem()));

        let mut scope = context.into_scope();
        let out: Variable = out.into();
        let three: Variable = 3u32.into();
        cpa!(scope, out = three);

        format!("{:?}", scope.operations)
    }
}
