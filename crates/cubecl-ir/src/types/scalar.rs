use pliron::{
    context::Context,
    derive::{pliron_type, type_interface_impl},
};

use crate::{
    ElemType, FloatKind, IntKind, UIntKind, aligned,
    interfaces::{AlignedType, MaybePackedType, ScalarType, SizedType, not_packed},
    scalar, sized,
};

#[pliron_type(
    name = "cube.int",
    format = "$width",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct IntType {
    pub width: usize,
}
scalar!(IntType);
not_packed!(IntType);

#[type_interface_impl]
impl AlignedType for IntType {
    fn align(&self, _ctx: &Context) -> usize {
        self.width / 8
    }
}

#[type_interface_impl]
impl SizedType for IntType {
    fn size(&self, _ctx: &Context) -> usize {
        self.width / 8
    }
}

#[type_interface_impl]
impl ScalarType for IntType {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        match self.width {
            8 => IntKind::I8,
            16 => IntKind::I16,
            32 => IntKind::I32,
            64 => IntKind::I64,
            _ => unreachable!("Unsupported bit width"),
        }
        .into()
    }
}

#[pliron_type(
    name = "cube.uint",
    format = "$width",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct UIntType {
    pub width: usize,
}
scalar!(UIntType);
not_packed!(UIntType);

#[type_interface_impl]
impl AlignedType for UIntType {
    fn align(&self, _ctx: &Context) -> usize {
        self.width / 8
    }
}

#[type_interface_impl]
impl SizedType for UIntType {
    fn size(&self, _ctx: &Context) -> usize {
        self.width / 8
    }
}

#[type_interface_impl]
impl ScalarType for UIntType {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        match self.width {
            8 => UIntKind::U8,
            16 => UIntKind::U16,
            32 => UIntKind::U32,
            64 => UIntKind::U64,
            _ => unreachable!("Unsupported bit width"),
        }
        .into()
    }
}

#[pliron_type(
    name = "cube.index",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct IndexType;
// May need only align_of::<u32>() for 32-bit addressing, but it's safe to give it more
aligned!(IndexType, align_of::<u64>());
sized!(IndexType, size_of::<u64>());
scalar!(IndexType);
not_packed!(IndexType);

#[type_interface_impl]
impl ScalarType for IndexType {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        ElemType::Index
    }
}

macro_rules! float_type {
    ($name: literal, $ty: ident, $kind: ident, $size: literal, $size_bits: expr) => {
        #[pliron_type(name = $name, format = "", generate_get = true, verifier = "succ")]
        #[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
        pub struct $ty;
        scalar!($ty);
        not_packed!($ty);
        aligned!($ty, $size);

        #[type_interface_impl]
        impl ScalarType for $ty {
            fn elem_type(&self, _ctx: &Context) -> ElemType {
                FloatKind::$kind.into()
            }
        }

        #[type_interface_impl]
        impl SizedType for $ty {
            fn size(&self, _ctx: &Context) -> usize {
                $size
            }

            fn size_bits(&self, _ctx: &Context) -> usize {
                $size_bits
            }
        }
    };
    ($name: literal, $ty: ident, $kind: ident, $size: literal) => {
        float_type!($name, $ty, $kind, $size, $size * 8);
    };
}

float_type!("cube.f64", Float64Type, F64, 8);
float_type!("cube.f32", Float32Type, F32, 4);
float_type!("cube.tf32", TFloat32Type, TF32, 4);
float_type!("cube.flex32", FloatFlex32Type, Flex32, 4);
float_type!("cube.f16", Float16Type, F16, 2);
float_type!("cube.bf16", BFloat16Type, BF16, 2);
float_type!("cube.ue8m0", Float8E8M0Type, UE8M0, 1);
float_type!("cube.e5m2", Float8E5M2Type, E5M2, 1);
float_type!("cube.e4m3", Float8E4M3Type, E4M3, 1);
float_type!("cube.e3m2", Float6E3M2Type, E3M2, 1);
float_type!("cube.e2m3", Float6E2M3Type, E2M3, 1);
float_type!("cube.e2m1", Float4E2M1Type, E2M1, 1, 4);

#[pliron_type(
    name = "cube.e2m1x2",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct Float4E2M1x2Type;
scalar!(Float4E2M1x2Type);
aligned!(Float4E2M1x2Type, 1);
sized!(Float4E2M1x2Type, 1);

#[type_interface_impl]
impl MaybePackedType for Float4E2M1x2Type {
    fn packing_factor(&self, _ctx: &Context) -> usize {
        2
    }
}

#[type_interface_impl]
impl ScalarType for Float4E2M1x2Type {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        FloatKind::E2M1x2.into()
    }
}

#[pliron_type(
    name = "cube.bool",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct BoolType;
aligned!(BoolType, 1);
scalar!(BoolType);
not_packed!(BoolType);

#[type_interface_impl]
impl ScalarType for BoolType {
    fn elem_type(&self, _ctx: &Context) -> ElemType {
        ElemType::Bool
    }
}
