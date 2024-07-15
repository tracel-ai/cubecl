use crate::frontend::{CubePrimitive, CubeType};
use crate::ir::Elem;
use crate::prelude::CubeContext;

use super::{init_expand_element, ExpandElement, ExpandElementTyped, Init, Vectorized};

// To be consistent with other primitive type.
/// Boolean type.
pub type Bool = bool;

/// Extension trait for [bool].
pub trait BoolOps {
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(_context: &mut CubeContext, value: bool) -> ExpandElementTyped<bool> {
        let var: ExpandElement = value.into();
        var.into()
    }
}

impl BoolOps for Bool { }

impl CubeType for bool {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for bool {
    fn as_elem() -> Elem {
        Elem::Bool
    }
}

impl Init for ExpandElementTyped<bool> {
    fn init(self, context: &mut crate::prelude::CubeContext) -> Self {
        ExpandElementTyped::new(init_expand_element(context, self))
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
