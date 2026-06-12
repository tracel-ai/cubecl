use pliron::{
    context::{Context, Ptr},
    derive::{pliron_type, type_interface_impl},
    r#type::{Type, TypeObj},
};

use crate::{
    ElemType, FloatKind, IntKind, StorageType, UIntKind,
    interfaces::{
        AlignedType, ScalarType, ScalarizableType, SizedType, aligned, not_packed, scalar, sized,
    },
};

#[pliron_type(
    name = "cube.int",
    format = "`i` $width",
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
    fn storage_type(&self, _ctx: &Context) -> StorageType {
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

#[type_interface_impl]
impl ScalarizableType for IntType {
    fn scalar_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        self.get_self_ptr(ctx)
    }
}

#[pliron_type(
    name = "cube.uint",
    format = "`u` $width",
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
    fn storage_type(&self, _ctx: &Context) -> StorageType {
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

#[type_interface_impl]
impl ScalarizableType for UIntType {
    fn scalar_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        self.get_self_ptr(ctx)
    }
}

#[pliron_type(
    name = "cube.index",
    format = "`usize`",
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
impl ScalarizableType for IndexType {
    fn scalar_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        self.get_self_ptr(ctx)
    }
}

#[pliron_type(
    name = "cube.float",
    format = "$encoding",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct FloatType {
    pub encoding: FloatKind,
}
scalar!(FloatType);
not_packed!(FloatType);

#[type_interface_impl]
impl AlignedType for FloatType {
    fn align(&self, _ctx: &Context) -> usize {
        self.encoding.size()
    }
}

#[type_interface_impl]
impl SizedType for FloatType {
    fn size(&self, _ctx: &Context) -> usize {
        self.encoding.size()
    }
}

#[type_interface_impl]
impl ScalarType for FloatType {
    fn storage_type(&self, _ctx: &Context) -> StorageType {
        self.encoding.into()
    }
}

#[type_interface_impl]
impl ScalarizableType for FloatType {
    fn scalar_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        self.get_self_ptr(ctx)
    }
}

#[pliron_type(
    name = "cube.bool",
    format = "`bool`",
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
    fn storage_type(&self, _ctx: &Context) -> StorageType {
        ElemType::Bool.into()
    }
}

#[type_interface_impl]
impl ScalarizableType for BoolType {
    fn scalar_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        self.get_self_ptr(ctx)
    }
}
