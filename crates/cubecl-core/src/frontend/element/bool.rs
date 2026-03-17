use cubecl_ir::{ConstantValue, ManagedVariable, Scope, StorageType, Type};

use crate::{
    frontend::{CubePrimitive, CubeType},
    prelude::Scalar,
};
use crate::{ir::ElemType, prelude::Const};

use super::{IntoRuntime, NativeAssign, NativeExpand};

/// Extension trait for [bool].
pub trait BoolOps {
    #[allow(clippy::new_ret_no_self)]
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(_scope: &mut Scope, value: NativeExpand<bool>) -> NativeExpand<bool> {
        ManagedVariable::Plain(ElemType::Bool.constant(value.expand.as_const().unwrap())).into()
    }
}

impl BoolOps for bool {}

impl CubeType for bool {
    type ExpandType = NativeExpand<Self>;
}

impl Scalar for bool {}
impl CubePrimitive for bool {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

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
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl NativeAssign for bool {}
