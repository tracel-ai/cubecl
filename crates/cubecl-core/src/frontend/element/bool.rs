use cubecl_ir::{ExpandElement, Scope};

use crate::frontend::{CubePrimitive, CubeType};
use crate::ir::Elem;

use super::{init_expand_element, ExpandElementBaseInit, ExpandElementTyped, Init, IntoRuntime};

/// Extension trait for [bool].
pub trait BoolOps {
    #[allow(clippy::new_ret_no_self)]
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(
        _context: &mut Scope,
        value: ExpandElementTyped<bool>,
    ) -> ExpandElementTyped<bool> {
        ExpandElement::Plain(Elem::Bool.from_constant(*value.expand)).into()
    }
}

impl BoolOps for bool {}

impl CubeType for bool {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for bool {
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Bool)
    }
}

impl IntoRuntime for bool {
    fn __expand_runtime_method(self, context: &mut Scope) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        Init::init(expand, context)
    }
}

impl ExpandElementBaseInit for bool {
    fn init_elem(context: &mut Scope, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}
