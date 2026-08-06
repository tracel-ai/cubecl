use pliron::derive::{pliron_type, type_interface_impl};

use crate::{
    AddressSpace, aligned,
    interfaces::{
        AlignedType, HasElementType, IndexableType, MaybePackedType, MaybeVectorizedType,
        ScalarizableType, SizedType, TypedExt,
    },
    prelude::*,
    sized,
};

pub mod aggregate;
pub mod barrier;
pub mod cuda;
pub mod matrix;
pub mod scalar;
pub mod spirv;

pub use matrix::{MatrixIdent, MatrixLayout, MatrixScope, MatrixShape};

#[pliron_type(
    name = "vector.vector",
    format = "`<` $vectorization ` x ` $inner `>`",
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

#[type_interface_impl]
impl HasElementType for VectorType {
    fn element_type(&self, ctx: &Context) -> Option<TypeHandle> {
        Some(self.get_self_handle(ctx))
    }
}

#[pliron_type(
    name = "atomic.atomic",
    format = "`<` $inner `>`",
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
    fn try_vector_size(&self, ctx: &Context) -> Option<usize> {
        self.inner.try_get_vector_size(ctx)
    }
}

#[type_interface_impl]
impl AlignedType for AtomicType {
    fn align(&self, ctx: &Context) -> usize {
        self.inner.align(ctx)
    }
}

#[type_interface_impl]
impl SizedType for AtomicType {
    fn size(&self, ctx: &Context) -> usize {
        self.inner.size(ctx)
    }
}

#[type_interface_impl]
impl HasElementType for AtomicType {
    fn element_type(&self, ctx: &Context) -> Option<TypeHandle> {
        type_cast::<dyn HasElementType>(&*self.inner.deref(ctx))?.element_type(ctx)
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
    fn try_vector_size(&self, ctx: &Context) -> Option<usize> {
        self.inner.try_get_vector_size(ctx)
    }
}

#[type_interface_impl]
impl MaybePackedType for PointerType {
    fn packing_factor(&self, ctx: &Context) -> usize {
        self.inner.packing_factor(ctx)
    }
}

#[type_interface_impl]
impl HasElementType for PointerType {
    fn element_type(&self, ctx: &Context) -> Option<TypeHandle> {
        type_cast::<dyn HasElementType>(&*self.inner.deref(ctx))?.element_type(ctx)
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
    fn try_vector_size(&self, ctx: &Context) -> Option<usize> {
        self.inner.try_get_vector_size(ctx)
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

#[type_interface_impl]
impl MaybePackedType for ArrayType {
    fn packing_factor(&self, ctx: &Context) -> usize {
        self.inner.packing_factor(ctx)
    }
}

#[type_interface_impl]
impl HasElementType for ArrayType {
    fn element_type(&self, ctx: &Context) -> Option<TypeHandle> {
        type_cast::<dyn HasElementType>(&*self.inner.deref(ctx))?.element_type(ctx)
    }
}

/// Raw byte array.
/// Separate to mark it as semantically opaque and not valid as an input to ops that take a normal
/// array.
#[pliron_type(
    name = "cube.bytes",
    format = "",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
pub struct BytesType;
aligned!(BytesType, align_of::<u8>());

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
    fn try_vector_size(&self, ctx: &Context) -> Option<usize> {
        self.inner.try_get_vector_size(ctx)
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

#[type_interface_impl]
impl MaybePackedType for RuntimeArrayType {
    fn packing_factor(&self, ctx: &Context) -> usize {
        self.inner.packing_factor(ctx)
    }
}

#[type_interface_impl]
impl HasElementType for RuntimeArrayType {
    fn element_type(&self, ctx: &Context) -> Option<TypeHandle> {
        type_cast::<dyn HasElementType>(&*self.inner.deref(ctx))?.element_type(ctx)
    }
}
