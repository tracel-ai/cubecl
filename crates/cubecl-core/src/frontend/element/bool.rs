use cubecl_ir::{ConstantValue, ExpandElement, Scope, StorageType};

use crate::frontend::{CubePrimitive, CubeType};
use crate::ir::ElemType;

use super::{ExpandElementAssign, ExpandElementTyped, IntoMut, IntoRuntime};

/// Extension trait for [bool].
pub trait BoolOps {
    #[allow(clippy::new_ret_no_self)]
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(
        _scope: &mut Scope,
        value: ExpandElementTyped<bool>,
    ) -> ExpandElementTyped<bool> {
        ExpandElement::Plain(ElemType::Bool.constant(value.expand.as_const().unwrap())).into()
    }
}

impl BoolOps for bool {}

impl CubeType for bool {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for bool {
    fn as_type_native() -> Option<StorageType> {
        Some(StorageType::Scalar(ElemType::Bool))
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Bool(value) = value else {
            unreachable!()
        };
        value
    }
}

impl IntoRuntime for bool {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        IntoMut::into_mut(expand, scope)
    }
}

impl ExpandElementAssign for bool {}
