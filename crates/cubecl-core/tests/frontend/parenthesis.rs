use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn parenthesis<T: Numeric>(x: T, y: T, z: T) -> T {
    x * (y + z)
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Item, Variable},
    };

    type ElemType = f32;

    #[test]
    fn cube_parenthesis_priority_test() {
        let mut context = CubeContext::root();

        let x = context.create_local(Item::new(ElemType::as_elem()));
        let y = context.create_local(Item::new(ElemType::as_elem()));
        let z = context.create_local(Item::new(ElemType::as_elem()));

        parenthesis::expand::<ElemType>(&mut context, x.into(), y.into(), z.into());
        let scope = context.into_scope();

        assert_eq!(format!("{:?}", scope.operations), inline_macro_ref());
    }

    fn inline_macro_ref() -> String {
        let mut context = CubeContext::root();
        let item = Item::new(ElemType::as_elem());
        let x = context.create_local(item);
        let y = context.create_local(item);
        let z = context.create_local(item);

        let mut scope = context.into_scope();
        let x: Variable = x.into();
        let y: Variable = y.into();
        let z: Variable = z.into();

        cpa!(scope, y = y + z);
        cpa!(scope, x = x * y);

        format!("{:?}", scope.operations)
    }
}
