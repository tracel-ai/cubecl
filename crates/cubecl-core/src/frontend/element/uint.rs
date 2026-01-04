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
