use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn tuple_const() -> (UInt, UInt) {
    let x = UInt::new(0);
    let y = UInt::new(1);
    (x, y)
}

mod tests {
    use super::*;
    use cubecl_core::{
        cpa,
        ir::{Elem, Item, Operation, Variable},
    };

    #[test]
    fn cube_tuple_const_test() {
        let mut context = CubeContext::root();

        tuple_const::__expand(&mut context);
        let scope = context.into_scope();

        assert_eq!(scope.operations, inline_macro_ref_tuple_const());
    }

    fn inline_macro_ref_tuple_const() -> Vec<Operation> {
        let context = CubeContext::root();

        let mut scope = context.into_scope();
        let x = scope.create_local(Item::new(Elem::UInt));
        let y = scope.create_local(Item::new(Elem::UInt));

        let zero: Variable = 0u32.into();
        let one: Variable = 1u32.into();

        cpa!(scope, x = zero);
        cpa!(scope, y = one);

        scope.operations
    }
}
