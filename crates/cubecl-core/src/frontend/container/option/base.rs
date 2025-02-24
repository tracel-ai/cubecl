use crate::frontend::CubeType;
use crate::prelude::CubeDebug;
use cubecl_ir::Scope;

impl<T: CubeType> CubeType for Option<T>
where
    T::ExpandType: Clone,
{
    type ExpandType = Option<T::ExpandType>;
}

impl<T: CubeDebug> CubeDebug for Option<T> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        if let Some(value) = &self {
            value.set_debug_name(scope, name)
        }
    }
}

// impl<T: ExpandElementBaseInit> ExpandElementBaseInit for Option<T> {
//     fn init_elem(_scope: &mut crate::ir::Scope, elem: ExpandElement) -> ExpandElement {
//         // The type can't be deeply cloned/copied.
//         elem
//     }
// }
