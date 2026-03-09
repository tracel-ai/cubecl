use cubecl_ir::{ConstantValue, ExpandElement, Scope, StorageType, Type};

use crate::ir::ElemType;
use crate::{
    frontend::{CubePrimitive, CubeType},
    prelude::Scalar,
};

use super::{ExpandElementAssign, ExpandElementTyped, IntoRuntime};

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

impl Scalar for bool {}
impl CubePrimitive for bool {
    type Scalar = Self;

    fn as_type_native() -> Option<Type> {
        Some(StorageType::Scalar(ElemType::Bool).into())
    }

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Bool(value) = value else {
            unreachable!()
        };
        value
    }
}

impl IntoRuntime for bool {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> ExpandElementTyped<Self> {
        self.into()
    }
}

impl ExpandElementAssign for bool {}
