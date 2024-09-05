use crate::frontend::{CubePrimitive, CubeType};
use crate::ir::Elem;
use crate::prelude::{ComptimeType, CubeContext};

use super::{
    init_expand_element, ExpandElement, ExpandElementBaseInit, ExpandElementTyped, Vectorized,
};

// To be consistent with other primitive type.
/// Boolean type.
pub type Bool = bool;

/// Extension trait for [bool].
pub trait BoolOps {
    #[allow(clippy::new_ret_no_self)]
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(
        _context: &mut CubeContext,
        value: ExpandElementTyped<bool>,
    ) -> ExpandElementTyped<bool> {
        ExpandElement::Plain(Elem::Bool.from_constant(*value.expand)).into()
    }
}

impl BoolOps for Bool {}

impl ComptimeType for Bool {
    fn into_expand(self) -> Self::ExpandType {
        ExpandElementTyped::new(self.into())
    }
}

impl CubeType for bool {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for Bool {
    fn as_elem() -> Elem {
        Elem::Bool
    }
}

impl ExpandElementBaseInit for bool {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl Vectorized for bool {
    fn vectorization_factor(&self) -> crate::prelude::UInt {
        todo!()
    }

    fn vectorize(self, _factor: crate::prelude::UInt) -> Self {
        todo!()
    }
}
