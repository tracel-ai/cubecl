use cubecl_ir::{ConstantValue, ExpandElement, Scope, StorageType, UIntKind};

use crate::frontend::{CubePrimitive, CubeType, Numeric};
use crate::ir::ElemType;

use super::{
    ExpandElementIntoMut, ExpandElementTyped, Int, IntoMut, IntoRuntime, into_mut_expand_element,
    into_runtime_expand_element,
};

macro_rules! declare_uint {
    ($primitive:ident, $kind:ident) => {
        impl CubeType for $primitive {
            type ExpandType = ExpandElementTyped<Self>;
        }

        impl CubePrimitive for $primitive {
            fn as_type_native() -> Option<StorageType> {
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
            fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
                let elem: ExpandElementTyped<Self> = self.into();
                into_runtime_expand_element(scope, elem).into()
            }
        }

        impl IntoMut for $primitive {
            fn into_mut(self, _scope: &mut Scope) -> Self {
                self
            }
        }

        impl ExpandElementIntoMut for $primitive {
            fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
                into_mut_expand_element(scope, elem)
            }
        }

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
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for usize {
    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::UInt(value) = value else {
            unreachable!()
        };
        value as usize
    }

    fn as_type(scope: &Scope) -> StorageType {
        scope.resolve_type::<Self>().expect("Type to be registered")
    }
}

impl IntoRuntime for usize {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl IntoMut for usize {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl ExpandElementIntoMut for usize {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

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
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for isize {
    fn from_const_value(value: ConstantValue) -> Self {
        let ConstantValue::Int(value) = value else {
            unreachable!()
        };
        value as isize
    }

    fn as_type(scope: &Scope) -> StorageType {
        scope.resolve_type::<Self>().expect("Type to be registered")
    }
}

impl IntoRuntime for isize {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl IntoMut for isize {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl ExpandElementIntoMut for isize {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

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
