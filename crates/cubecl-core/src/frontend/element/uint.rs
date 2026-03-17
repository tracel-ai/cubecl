use cubecl_ir::{ConstantValue, Scope, Type, UIntKind};

use crate::ir::ElemType;
use crate::prelude::*;

use super::{IntoMut, IntoRuntime, NativeAssign, NativeExpand};

macro_rules! declare_uint {
    ($primitive:ident, $kind:ident) => {
        impl CubeType for $primitive {
            type ExpandType = NativeExpand<Self>;
        }

        impl Scalar for $primitive {}
        impl CubePrimitive for $primitive {
            type Scalar = Self;
            type Size = Const<1>;
            type WithScalar<S: Scalar> = S;

            fn as_type_native() -> Option<Type> {
                Some(ElemType::UInt(UIntKind::$kind).into())
            }

            fn from_const_value(value: ConstantValue) -> Self {
                let ConstantValue::UInt(value) = value else {
                    unreachable!()
                };
                value as $primitive
            }
        }

        impl IntoRuntime for $primitive {
            fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
                self.into()
            }
        }

        impl IntoMut for $primitive {
            fn into_mut(self, _scope: &mut Scope) -> Self {
                self
            }
        }

        impl NativeAssign for $primitive {}

        impl Numeric for $primitive {
            fn min_value() -> Self {
                $primitive::MIN
            }
            fn max_value() -> Self {
                $primitive::MAX
            }
        }

        impl Int for $primitive {
            const BITS: u32 = $primitive::BITS;

            fn new(val: i64) -> Self {
                val as $primitive
            }
        }
    };
}

declare_uint!(u8, U8);
declare_uint!(u16, U16);
declare_uint!(u32, U32);
declare_uint!(u64, U64);

impl CubeType for usize {
    type ExpandType = NativeExpand<Self>;
}

impl Scalar for usize {}
impl CubePrimitive for usize {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::UInt(value) = value else {
            unreachable!()
        };
        value as usize
    }

    fn as_type(scope: &Scope) -> Type {
        Type::new(scope.resolve_type::<Self>().expect("Type to be registered"))
    }
}

impl IntoRuntime for usize {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl IntoMut for usize {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl NativeAssign for usize {}

impl Numeric for usize {
    fn min_value() -> Self {
        usize::MIN
    }
    fn max_value() -> Self {
        // Stay in safe range. Should use runtime version taking scope for correct value.
        u32::MAX as usize
    }
}

impl Int for usize {
    const BITS: u32 = usize::BITS;

    fn new(val: i64) -> Self {
        val as usize
    }
}

impl CubeType for isize {
    type ExpandType = NativeExpand<Self>;
}

impl Scalar for isize {}
impl CubePrimitive for isize {
    type Scalar = Self;
    type Size = Const<1>;
    type WithScalar<S: Scalar> = S;

    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Int(value) = value else {
            unreachable!()
        };
        value as isize
    }

    fn as_type(scope: &Scope) -> Type {
        Type::new(scope.resolve_type::<Self>().expect("Type to be registered"))
    }
}

impl IntoRuntime for isize {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> NativeExpand<Self> {
        self.into()
    }
}

impl IntoMut for isize {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl NativeAssign for isize {}

impl Numeric for isize {
    fn min_value() -> Self {
        i32::MIN as isize
    }
    fn max_value() -> Self {
        // Stay in safe range. Should use runtime version taking scope for correct value.
        i32::MAX as isize
    }
}

impl Int for isize {
    const BITS: u32 = isize::BITS;

    fn new(val: i64) -> Self {
        val as isize
    }
}
