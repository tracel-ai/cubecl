use alloc::vec;
use core::ops::{Deref, DerefMut};

use alloc::boxed::Box;

use crate::{self as cubecl, unexpanded};
use cubecl::prelude::*;
use cubecl_ir::{
    OpInserter, SliceMetadata, VectorSize,
    dialect::{
        branch::RangeLoopOp,
        general::{AggregateConstructOp, ReinterpretCastOp},
    },
    interfaces::TypedExt,
    pliron::{
        builtin::op_interfaces::OneResultInterface,
        context::{Context, Ptr},
        printable::Printable,
        r#type::{TypeObj, Typed},
        value::Value,
    },
    types::{ArrayType, PointerType, RuntimeArrayType, VectorType, scalar::IndexType},
};

pub type SliceExpand<T> = NativeExpand<[T]>;

#[derive(Clone, Copy)]
pub struct ReadOnly;
#[derive(Clone, Copy)]
pub struct ReadWrite;

pub trait SliceVisibility: Clone + Copy + Send + Sync + 'static {}

impl SliceVisibility for ReadOnly {}

impl SliceVisibility for ReadWrite {}

impl<E: CubePrimitive> SliceExpand<E> {
    pub fn __extract_list(&self, scope: &Scope) -> Value {
        scope.extract_field(self.value(scope), SliceMetadata::LIST)
    }

    pub fn __extract_offset(&self, scope: &Scope) -> NativeExpand<usize> {
        let field = scope.extract_field(self.value(scope), SliceMetadata::OFFSET);
        field.into()
    }

    pub fn __extract_length(&self, scope: &Scope) -> NativeExpand<usize> {
        let field = scope.extract_field(self.value(scope), SliceMetadata::LENGTH);
        field.into()
    }
}

pub trait SliceVectorExt<E: Scalar, N: Size> {
    fn with_vector_size<N2: Size>(&self) -> &[Vector<E, N2>] {
        unexpanded!()
    }
    fn with_vector_size_mut<N2: Size>(&mut self) -> &mut [Vector<E, N2>] {
        unexpanded!()
    }
    fn __expand_with_vector_size<'infer, N2: Size>(
        scope: &Scope,
        this: &'infer SliceExpand<Vector<E, N>>,
    ) -> &'infer SliceExpand<Vector<E, N2>> {
        this.__expand_with_vector_size_method(scope)
    }
    fn __expand_with_vector_size_mut<'infer, N2: Size>(
        scope: &Scope,
        this: &'infer mut SliceExpand<Vector<E, N>>,
    ) -> &'infer mut SliceExpand<Vector<E, N2>> {
        this.__expand_with_vector_size_mut_method(scope)
    }
}

impl<E: Scalar, N: Size> SliceVectorExt<E, N> for [Vector<E, N>] {}
impl<E: Scalar, N: Size> SliceExpand<Vector<E, N>> {
    pub fn __expand_with_vector_size_method<'infer, N2: Size>(
        &'infer self,
        scope: &Scope,
    ) -> &'infer SliceExpand<Vector<E, N2>> {
        let slice = self.with_vector_size_inner::<N2>(scope);
        scope.create_kernel_ref(slice)
    }

    pub fn __expand_with_vector_size_mut_method<'infer, N2: Size>(
        &'infer mut self,
        scope: &Scope,
    ) -> &'infer mut SliceExpand<Vector<E, N2>> {
        let slice = self.with_vector_size_inner::<N2>(scope);
        scope.create_kernel_ref(slice)
    }
}

impl<E: Scalar, N: Size> SliceExpand<Vector<E, N>> {
    fn with_vector_size_inner<N2: Size>(&self, scope: &Scope) -> SliceExpand<Vector<E, N2>> {
        let vector_size = N2::__expand_value(scope);
        let list = self.__extract_list(scope);

        let length = self.__extract_length(scope);
        let offset = self.__extract_offset(scope);

        let current = list.vector_size(&scope.ctx());

        if vector_size == current {
            return self.expand.into();
        }

        let new_ptr_ty = change_list_vectorization(&mut scope.ctx_mut(), list, vector_size);
        let reinterpret = ReinterpretCastOp::new(&mut scope.ctx_mut(), new_ptr_ty, list);
        scope.register(&reinterpret);
        let new_ptr = reinterpret.get_result(&scope.ctx());

        if current < vector_size {
            let ratio = vector_size / current;
            let offset = offset.__expand_div_method(scope, ratio.into());
            let length = length.__expand_div_method(scope, ratio.into());
            from_raw_parts(scope, new_ptr, offset, length)
        } else {
            let ratio = current / vector_size;
            let offset = offset.__expand_mul_method(scope, ratio.into());
            let length = length.__expand_mul_method(scope, ratio.into());
            from_raw_parts(scope, new_ptr, offset, length)
        }
    }
}

// This is really annoying but does have a lot more checks for invariants than before
fn change_list_vectorization(ctx: &mut Context, list: Value, new_vec: usize) -> Ptr<TypeObj> {
    let current_vec = list.vector_size(ctx);
    let ty = list.get_type(ctx);
    let PointerType {
        inner,
        address_space,
    } = {
        let ty = ty.deref(ctx);
        *ty.downcast_ref().unwrap()
    };
    let (arr, runtime_arr) = {
        let list_ty = inner.deref(ctx);
        let arr = list_ty.downcast_ref::<ArrayType>().copied();
        let runtime_arr = list_ty.downcast_ref::<RuntimeArrayType>().copied();
        (arr, runtime_arr)
    };
    if let Some(ArrayType { inner, length }) = arr {
        let new_length = length * current_vec / new_vec;
        let scalar_ty = inner.scalar_ty(ctx);
        let new_vector_ty = if new_vec > 1 {
            VectorType::get(ctx, scalar_ty, new_vec).into()
        } else {
            scalar_ty
        };
        let new_arr_ty = ArrayType::get(ctx, new_vector_ty, new_length).into();
        let new_ptr_ty = PointerType::get(ctx, new_arr_ty, address_space);
        return new_ptr_ty.into();
    }
    if let Some(RuntimeArrayType { inner }) = runtime_arr {
        let scalar_ty = inner.scalar_ty(ctx);
        let new_vector_ty = if new_vec > 1 {
            VectorType::get(ctx, scalar_ty, new_vec).into()
        } else {
            scalar_ty
        };
        let new_arr_ty = RuntimeArrayType::get(ctx, new_vector_ty).into();
        let new_ptr_ty = PointerType::get(ctx, new_arr_ty, address_space);
        return new_ptr_ty.into();
    }
    unreachable!("Should be static or dynamic array")
}

pub trait SliceExt<E: CubePrimitive> {
    /// Returns the same slice, but with the type reinterpreted as `Vector`.
    /// Preserves existing vector size of the primitive.
    fn as_vectorized(&self) -> &[Vector<E::Scalar, E::Size>] {
        unexpanded!()
    }

    /// Returns the same slice, but with the type reinterpreted as `Vector`.
    /// Preserves existing vector size of the primitive.
    fn as_vectorized_mut(&mut self) -> &mut [Vector<E::Scalar, E::Size>] {
        unexpanded!()
    }

    /// Downcast the slice to the given type and panic if the type isn't the same.
    ///
    /// This function should only be used to satisfy the Rust type system, when two generic
    /// types are supposed to be the same.
    fn downcast<T: CubePrimitive>(&self) -> &[T] {
        unexpanded!()
    }

    /// Downcast the slice to the given type and panic if the type isn't the same.
    ///
    /// This function should only be used to satisfy the Rust type system, when two generic
    /// types are supposed to be the same.
    fn downcast_mut<T: CubePrimitive>(&mut self) -> &mut [T] {
        unexpanded!()
    }

    /// Unsafely downcast the slice to the given type and panic if the type isn't the same.
    ///
    /// # Safety
    /// This function converts unsafely, and should only be used for temporary storage with a dummy
    /// type (i.e. `ReinterpretSlice`)
    unsafe fn downcast_unchecked<T: CubePrimitive>(&self) -> &[T] {
        unexpanded!()
    }

    /// Unsafely downcast the slice to the given type and panic if the type isn't the same.
    ///
    /// # Safety
    /// This function converts unsafely, and should only be used for temporary storage with a dummy
    /// type (i.e. `ReinterpretSlice`)
    unsafe fn downcast_mut_unchecked<T: CubePrimitive>(&mut self) -> &mut [T] {
        unexpanded!()
    }

    /// Unsafely cast an immutable slice to a mutable one.
    ///
    /// # Safety
    /// This is safe in practice, but breaks semantics. Should only be used if absolutely necessary.
    /// May cause problems if an immutable input is reinterpreted as mutable.
    #[allow(clippy::mut_from_ref)]
    unsafe fn as_mut_unchecked(&self) -> &mut [E] {
        unexpanded!()
    }

    /// Convert the slice to a start pointer, without any bounds checks.
    ///
    /// # Safety
    /// See [`get_unchecked`]([E]::get_unchecked)
    unsafe fn as_ptr_unchecked(&self) -> *const E {
        unexpanded!()
    }

    /// Convert the slice to a mutable start pointer, without any bounds checks.
    ///
    /// # Safety
    /// See [`get_unchecked_mut`]([E]::get_unchecked_mut)
    unsafe fn as_mut_ptr_unchecked(&mut self) -> *mut E {
        unexpanded!()
    }

    /// Convert to an owned boxed slice. This is very unsafe as it completely erases the lifetime.
    /// Only use it for global kernel inputs, which have a static lifetime.
    ///
    /// # Safety
    /// Erases the lifetime. Only use when an owned representation is absolutely needed.
    unsafe fn as_boxed_unchecked(&self) -> Box<[E]> {
        unexpanded!()
    }

    fn __expand_as_vectorized<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer SliceExpand<Vector<E::Scalar, E::Size>>;

    fn __expand_as_vectorized_mut<'infer>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut SliceExpand<Vector<E::Scalar, E::Size>>;

    fn __expand_downcast<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer SliceExpand<T>;

    fn __expand_downcast_mut<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut SliceExpand<T>;

    fn __expand_downcast_unchecked<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer SliceExpand<T>;

    fn __expand_downcast_mut_unchecked<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut SliceExpand<T>;

    fn __expand_as_ptr(scope: &Scope, this: &SliceExpand<E>) -> *const NativeExpand<E>;

    fn __expand_as_mut_ptr(scope: &Scope, this: &mut SliceExpand<E>) -> *mut NativeExpand<E>;

    #[doc(hidden)]
    unsafe fn __expand_as_ptr_unchecked(
        scope: &Scope,
        this: &SliceExpand<E>,
    ) -> *const NativeExpand<E>;

    #[doc(hidden)]
    unsafe fn __expand_as_mut_ptr_unchecked(
        scope: &Scope,
        this: &mut SliceExpand<E>,
    ) -> *mut NativeExpand<E>;

    #[allow(clippy::mut_from_ref)]
    #[doc(hidden)]
    unsafe fn __expand_as_mut_unchecked<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer mut SliceExpand<E>;

    #[doc(hidden)]
    unsafe fn __expand_as_boxed_unchecked(
        scope: &Scope,
        this: &SliceExpand<E>,
    ) -> NativeExpand<Box<[E]>>;
}

impl<E: CubePrimitive> SliceExt<E> for [E] {
    fn __expand_as_vectorized<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer SliceExpand<Vector<E::Scalar, E::Size>> {
        this.__expand_as_vectorized_method(scope)
    }

    fn __expand_as_vectorized_mut<'infer>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut SliceExpand<Vector<E::Scalar, E::Size>> {
        this.__expand_as_vectorized_mut_method(scope)
    }

    fn __expand_downcast<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer SliceExpand<T> {
        this.__expand_downcast_method::<T>(scope)
    }

    fn __expand_downcast_mut<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut SliceExpand<T> {
        this.__expand_downcast_mut_method::<T>(scope)
    }

    fn __expand_downcast_unchecked<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer SliceExpand<T> {
        this.__expand_downcast_unchecked_method::<T>(scope)
    }

    fn __expand_downcast_mut_unchecked<'infer, T: CubePrimitive>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut SliceExpand<T> {
        this.__expand_downcast_mut_unchecked_method::<T>(scope)
    }

    fn __expand_as_ptr(scope: &Scope, this: &SliceExpand<E>) -> *const NativeExpand<E> {
        this.__expand_as_ptr_method(scope)
    }

    fn __expand_as_mut_ptr(scope: &Scope, this: &mut SliceExpand<E>) -> *mut NativeExpand<E> {
        this.__expand_as_mut_ptr_method(scope)
    }

    unsafe fn __expand_as_ptr_unchecked(
        scope: &Scope,
        this: &SliceExpand<E>,
    ) -> *const NativeExpand<E> {
        unsafe { this.__expand_as_ptr_unchecked_method(scope) }
    }

    unsafe fn __expand_as_mut_ptr_unchecked(
        scope: &Scope,
        this: &mut SliceExpand<E>,
    ) -> *mut NativeExpand<E> {
        unsafe { this.__expand_as_mut_ptr_unchecked_method(scope) }
    }

    unsafe fn __expand_as_mut_unchecked<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer mut SliceExpand<E> {
        this.__expand_as_mut_unchecked_method(scope)
    }

    unsafe fn __expand_as_boxed_unchecked(
        scope: &Scope,
        this: &SliceExpand<E>,
    ) -> NativeExpand<Box<[E]>> {
        unsafe { this.__expand_as_boxed_unchecked_method(scope) }
    }
}

impl<E: CubePrimitive> SliceExpand<E> {
    pub fn __expand_as_vectorized_method(
        &self,
        _: &Scope,
    ) -> &SliceExpand<Vector<E::Scalar, E::Size>> {
        unsafe { self.as_type_ref_unchecked() }
    }

    pub fn __expand_as_vectorized_mut_method(
        &mut self,
        _: &Scope,
    ) -> &mut SliceExpand<Vector<E::Scalar, E::Size>> {
        unsafe { self.as_type_mut_unchecked() }
    }

    pub fn __expand_downcast_method<T: CubePrimitive>(&self, scope: &Scope) -> &SliceExpand<T> {
        let ty_t = T::__expand_as_type(scope);
        let ty_e = E::__expand_as_type(scope);
        if ty_t != ty_e && !is_tf32_cast::<E, T>(scope) && !is_flex32_cast::<E, T>(scope) {
            panic!(
                "Downcast should only be used to satisfy the Rust type system.
Expected types to be the same, got [{}, {}]",
                ty_t.disp(&scope.ctx()),
                ty_e.disp(&scope.ctx())
            )
        }

        self.__expand_downcast_unchecked_method(scope)
    }

    pub fn __expand_downcast_mut_method<T: CubePrimitive>(
        &mut self,
        scope: &Scope,
    ) -> &mut SliceExpand<T> {
        if T::__expand_as_type(scope) != E::__expand_as_type(scope) && !is_tf32_cast::<E, T>(scope)
        {
            let ty_t = T::__expand_as_type(scope);
            let ty_e = E::__expand_as_type(scope);
            if ty_t != ty_e && !is_tf32_cast::<E, T>(scope) && !is_flex32_cast::<E, T>(scope) {
                panic!(
                    "Downcast should only be used to satisfy the Rust type system.
Expected types to be the same, got [{}, {}]",
                    ty_t.disp(&scope.ctx()),
                    ty_e.disp(&scope.ctx())
                )
            }
        }

        self.__expand_downcast_mut_unchecked_method(scope)
    }

    #[doc(hidden)]
    pub fn __expand_downcast_unchecked_method<T: CubePrimitive>(
        &self,
        _: &Scope,
    ) -> &SliceExpand<T> {
        unsafe { self.as_type_ref_unchecked() }
    }

    #[doc(hidden)]
    pub fn __expand_downcast_mut_unchecked_method<T: CubePrimitive>(
        &mut self,
        _: &Scope,
    ) -> &mut SliceExpand<T> {
        unsafe { self.as_type_mut_unchecked() }
    }

    pub fn __expand_as_ptr_method(&self, scope: &Scope) -> *const NativeExpand<E> {
        self.__expand_index_method(scope, NativeExpand::<usize>::from_lit(scope, 0))
    }

    pub fn __expand_as_mut_ptr_method(&mut self, scope: &Scope) -> *mut NativeExpand<E> {
        self.__expand_index_mut_method(scope, NativeExpand::<usize>::from_lit(scope, 0))
    }

    #[doc(hidden)]
    pub unsafe fn __expand_as_ptr_unchecked_method(&self, scope: &Scope) -> *const NativeExpand<E> {
        unsafe {
            self.__expand_get_unchecked_method(scope, NativeExpand::<usize>::from_lit(scope, 0))
        }
    }

    #[doc(hidden)]
    pub unsafe fn __expand_as_mut_ptr_unchecked_method(
        &mut self,
        scope: &Scope,
    ) -> *mut NativeExpand<E> {
        unsafe {
            self.__expand_get_unchecked_mut_method(scope, NativeExpand::<usize>::from_lit(scope, 0))
        }
    }

    pub fn __expand_as_mut_unchecked_method(&self, scope: &Scope) -> &mut SliceExpand<E> {
        scope.create_kernel_ref(self.expand.into())
    }

    #[doc(hidden)]
    pub unsafe fn __expand_as_boxed_unchecked_method(&self, _: &Scope) -> NativeExpand<Box<[E]>> {
        self.expand.into()
    }
}

pub fn from_raw_parts<E: CubePrimitive>(
    scope: &Scope,
    list: Value,
    offset: NativeExpand<usize>,
    length: NativeExpand<usize>,
) -> SliceExpand<E> {
    let ty = list.get_type(&scope.ctx());
    let offset = offset.read_value(scope);
    let length = length.read_value(scope);
    let op = AggregateConstructOp::new(&mut scope.ctx_mut(), ty, vec![list, offset, length]);
    scope.register(&op);
    op.get_result(&scope.ctx()).into()
}

impl<E: CubePrimitive> SliceExpand<E> {
    /// Get the length of the slice.
    pub fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__extract_length(scope)
    }
    /// Returns true if the slice is empty.
    pub fn is_empty(&self, scope: &Scope) -> NativeExpand<bool> {
        self.__extract_length(scope)
            .__expand_eq_method(scope, &0usize.into_expand(scope))
    }
}

impl<E: CubePrimitive> CubeType for [E] {
    type ExpandType = SliceExpand<E>;
}

impl<E: CubePrimitive> CubeType for Box<[E]> {
    type ExpandType = NativeExpand<Box<[E]>>;
}

impl<E> Deref for NativeExpand<Box<[E]>> {
    type Target = NativeExpand<[E]>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.as_type_ref_unchecked() }
    }
}

impl<E> DerefMut for NativeExpand<Box<[E]>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_type_mut_unchecked() }
    }
}

macro_rules! impl_expand_traits {
    ($generic: ident, $ty: ty) => {
        impl<$generic: CubePrimitive> AsMutExpand for $ty {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }

        impl<$generic: CubePrimitive> IntoMut for $ty {
            fn into_mut(self, _scope: &Scope) -> Self {
                self
            }
        }
    };
}

impl_expand_traits!(E, SliceExpand<E>);
impl_expand_traits!(E, NativeExpand<Box<[E]>>);

impl<'a, E: CubePrimitive> From<&'a NativeExpand<Box<[E]>>> for &'a NativeExpand<[E]> {
    fn from(value: &'a NativeExpand<Box<[E]>>) -> Self {
        unsafe { value.as_type_ref_unchecked() }
    }
}

impl<'a, E: CubePrimitive> From<&'a mut NativeExpand<Box<[E]>>> for &'a mut NativeExpand<[E]> {
    fn from(value: &'a mut NativeExpand<Box<[E]>>) -> Self {
        unsafe { value.as_type_mut_unchecked() }
    }
}

impl<E: CubePrimitive> SizedContainer<usize> for [E] {
    fn len(&self) -> usize {
        unexpanded!()
    }
}

impl<E: CubePrimitive> SizedContainerExpand<usize> for SliceExpand<E> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope)
    }
}

impl<E: CubePrimitive> Iterable for SliceExpand<E> {
    type Item = E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = IndexType::get(&scope.ctx());

        let start = scope.const_usize(0);
        let end = self.__extract_length(scope).value(scope);
        let step = scope.const_usize(1);

        let i = scope.create_local_mut(index_ty);
        let range_loop = RangeLoopOp::new(&mut scope.ctx_mut(), i, start, end, step, false);
        let loop_body = range_loop.loop_body(&scope.ctx());

        let child = scope.child(OpInserter::new_at_block_end(loop_body));

        let index = NativeExpand::new(i.into());
        let item = self
            .__expand_index_method(&child, index)
            .__expand_deref_method(&child);
        body(&child, item);

        scope.register(&range_loop);
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<'a, E: CubePrimitive> Iterable for &'a SliceExpand<E> {
    type Item = &'a E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = IndexType::get(&scope.ctx());

        let start = scope.const_usize(0);
        let end = self.__extract_length(scope).value(scope);
        let step = scope.const_usize(1);

        let i = scope.create_local_mut(index_ty);
        let range_loop = RangeLoopOp::new(&mut scope.ctx_mut(), i, start, end, step, false);
        let loop_body = range_loop.loop_body(&scope.ctx());

        let child = scope.child(OpInserter::new_at_block_end(loop_body));

        let index = NativeExpand::new(i.into());
        let item = self.__expand_index_method(&child, index);
        body(&child, item);

        scope.register(&range_loop);
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<'a, E: CubePrimitive> Iterable for &'a mut SliceExpand<E> {
    type Item = &'a mut E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = IndexType::get(&scope.ctx());

        let start = scope.const_usize(0);
        let end = self.__extract_length(scope).value(scope);
        let step = scope.const_usize(1);

        let i = scope.create_local_mut(index_ty);
        let range_loop = RangeLoopOp::new(&mut scope.ctx_mut(), i, start, end, step, false);
        let loop_body = range_loop.loop_body(&scope.ctx());

        let child = scope.child(OpInserter::new_at_block_end(loop_body));

        let index = NativeExpand::new(i.into());
        let item = self.__expand_index_mut_method(&child, index);
        body(&child, item);

        scope.register(&range_loop);
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<E: CubePrimitive> SliceExpand<E> {
    #[doc(hidden)]
    pub unsafe fn __expand_get_unchecked_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &NativeExpand<E> {
        read_offset::expand::<E>(scope, self, index, false)
    }

    #[doc(hidden)]
    pub unsafe fn __expand_get_unchecked_mut_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut NativeExpand<E> {
        write_offset::expand::<E>(scope, self, index, false)
    }
}

impl<E: CubePrimitive> IndexExpand<NativeExpand<usize>> for SliceExpand<E> {
    type Output = E::ExpandType;

    fn __expand_index_method(&self, scope: &Scope, index: NativeExpand<usize>) -> &Self::Output {
        read_offset::expand::<E>(scope, self, index, true)
    }
}

impl<E: CubePrimitive> IndexMutExpand<NativeExpand<usize>> for SliceExpand<E> {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut Self::Output {
        write_offset::expand::<E>(scope, self, index, true)
    }
}

impl_slice_ranges!(SliceExpand<E>);

impl<E: CubePrimitive> List<E> for [E] {}
impl<E: CubePrimitive> ListExpand<E> for SliceExpand<E> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope)
    }
}

impl<E: CubePrimitive> Vectorized for Box<[E]> {}
impl<E: CubePrimitive> Vectorized for [E] {}
impl<E: CubePrimitive> VectorizedExpand for SliceExpand<E> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.value(scope).vector_size(&scope.ctx())
    }
}
impl<E: CubePrimitive> VectorizedExpand for NativeExpand<Box<[E]>> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.value(scope).vector_size(&scope.ctx())
    }
}

mod read_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        slice: &SliceExpand<E>,
        index: NativeExpand<usize>,
        checked: bool,
    ) -> &'a <E as cubecl::prelude::CubeType>::ExpandType {
        let list = slice.__extract_list(scope);
        let offset = slice.__extract_offset(scope);
        let index = offset.__expand_add_method(scope, index);

        expand_index_native(scope, list, index, checked)
    }
}

mod write_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        slice: &SliceExpand<E>,
        index: <usize as CubeType>::ExpandType,
        checked: bool,
    ) -> &'a mut E::ExpandType {
        let list = slice.__extract_list(scope);
        let offset = slice.__extract_offset(scope);
        let index = offset.__expand_add_method(scope, index);

        expand_index_mut_native(scope, list, index, checked)
    }
}
