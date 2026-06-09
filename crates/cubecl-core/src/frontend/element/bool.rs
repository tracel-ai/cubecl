use cubecl_ir::{ConstantValue, Scope, StorageType, Type};

use crate::prelude::*;
use crate::{ir::ElemType, prelude::Const};

use super::{IntoRuntime, NativeAssign, NativeExpand};

/// Extension trait for [bool].
pub trait BoolOps {
    #[allow(clippy::new_ret_no_self)]
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(_scope: &Scope, value: NativeExpand<bool>) -> NativeExpand<bool> {
        ElemType::Bool
            .constant(value.expand.as_const().unwrap())
            .into()
    }
}

impl BoolOps for bool {}

impl CubeType for bool {
    type ExpandType = NativeExpand<Self>;
}

impl CubeDebug for bool {}
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
    fn __expand_runtime_method(self, _scope: &Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl IntoExpand for bool {
    type Expand = NativeExpand<bool>;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self.into()
    }
}

impl NativeAssign for bool {}
