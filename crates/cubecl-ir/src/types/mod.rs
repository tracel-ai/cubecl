use pliron::{
    derive::{pliron_type, type_interface_impl},
    r#type::type_cast,
};

use crate::{
    AddressSpace, StorageType,
    interfaces::{
        AlignedType, IndexableType, MaybePackedType, MaybeVectorizedType, ScalarType,
        ScalarizableType, SizedType, TypedExt, aligned, scalar, sized,
    },
    prelude::*,
};

pub mod aggregate;
pub mod barrier;
pub mod cuda;
pub mod matrix;
pub mod scalar;
pub mod spirv;

pub use matrix::{MatrixIdent, MatrixLayout, MatrixScope, MatrixShape};

#[pliron_type(
    name = "cube.packed",
    format = "$inner `x` $packing_factor",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct PackedType {
    pub inner: TypeHandle,
    pub packing_factor: usize,
}
scalar!(PackedType);

#[type_interface_impl]
impl ScalarType for PackedType {
    fn storage_type(&self, ctx: &Context) -> StorageType {
        let inner = self.inner.deref(ctx);
        let inner_scalar = type_cast::<dyn ScalarType>(&*inner).unwrap();
        let inner_ty = inner_scalar.storage_type(ctx).elem_type();
        StorageType::Packed(inner_ty, self.packing_factor)
    }
}

#[type_interface_impl]
impl MaybePackedType for PackedType {
    fn packing_factor(&self, _ctx: &Context) -> usize {
        self.packing_factor
    }
}

#[type_interface_impl]
impl AlignedType for PackedType {
    fn align(&self, ctx: &Context) -> usize {
        self.inner.align(ctx) * self.packing_factor
    }
}

#[type_interface_impl]
impl ScalarizableType for PackedType {
    fn scalar_type(&self, ctx: &Context) -> TypeHandle {
        self.get_self_handle(ctx)
    }
}

#[pliron_type(
    name = "cube.vector",
    format = "`<` $inner `, ` $vectorization `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct VectorType {
    pub inner: TypeHandle,
    pub vectorization: usize,
}

#[type_interface_impl]
impl MaybeVectorizedType for VectorType {
    fn vector_size(&self, _ctx: &Context) -> usize {
        self.vectorization
    }
}

#[type_interface_impl]
impl MaybePackedType for VectorType {
    fn packing_factor(&self, ctx: &Context) -> usize {
        self.inner.packing_factor(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for VectorType {
    fn align(&self, ctx: &Context) -> usize {
        self.inner.align(ctx) * self.vectorization
    }
}

#[type_interface_impl]
impl SizedType for VectorType {
    fn size(&self, ctx: &Context) -> usize {
        self.inner.size(ctx) * self.vectorization
    }
}

#[type_interface_impl]
impl ScalarizableType for VectorType {
    fn scalar_type(&self, _ctx: &Context) -> TypeHandle {
        self.inner
    }
}

#[pliron_type(
    name = "cube.atomic",
    format = "`atomic<` $inner `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct AtomicType {
    pub inner: TypeHandle,
}

#[type_interface_impl]
impl MaybeVectorizedType for AtomicType {
    fn vector_size(&self, ctx: &Context) -> usize {
        self.inner.vector_size(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for AtomicType {
    fn align(&self, ctx: &Context) -> usize {
        self.inner.align(ctx)
    }
}

#[pliron_type(
    name = "cube.ptr",
    format = "`<` $inner `, ` $address_space `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct PointerType {
    pub inner: TypeHandle,
    pub address_space: AddressSpace,
}
aligned!(PointerType, align_of::<u64>());
sized!(PointerType, size_of::<u64>());

#[type_interface_impl]
impl MaybeVectorizedType for PointerType {
    fn vector_size(&self, ctx: &Context) -> usize {
        self.inner.vector_size(ctx)
    }
}

#[type_interface_impl]
impl MaybePackedType for PointerType {
    fn packing_factor(&self, ctx: &Context) -> usize {
        self.inner.packing_factor(ctx)
    }
}

#[pliron_type(
    name = "cube.array",
    format = "`[` $inner `; ` $length `]`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct ArrayType {
    pub inner: TypeHandle,
    pub length: usize,
}

#[type_interface_impl]
impl MaybeVectorizedType for ArrayType {
    fn vector_size(&self, ctx: &Context) -> usize {
        self.inner.vector_size(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for ArrayType {
    fn align(&self, ctx: &Context) -> usize {
        self.inner.align(ctx)
    }
}

#[type_interface_impl]
impl SizedType for ArrayType {
    fn size(&self, ctx: &Context) -> usize {
        self.inner.size(ctx) * self.length
    }
}

#[type_interface_impl]
impl IndexableType for ArrayType {
    fn indexed_type(&self, _ctx: &Context) -> TypeHandle {
        self.inner
    }
}

#[pliron_type(
    name = "cube.runtime_array",
    format = "`[` $inner `]`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct RuntimeArrayType {
    pub inner: TypeHandle,
}

#[type_interface_impl]
impl MaybeVectorizedType for RuntimeArrayType {
    fn vector_size(&self, ctx: &Context) -> usize {
        self.inner.vector_size(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for RuntimeArrayType {
    fn align(&self, ctx: &Context) -> usize {
        self.inner.align(ctx)
    }
}

#[type_interface_impl]
impl IndexableType for RuntimeArrayType {
    fn indexed_type(&self, _ctx: &Context) -> TypeHandle {
        self.inner
    }
}
