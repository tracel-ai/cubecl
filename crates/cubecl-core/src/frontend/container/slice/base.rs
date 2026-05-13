use alloc::vec;
use core::ops::{Deref, DerefMut};

use alloc::boxed::Box;

use crate::{self as cubecl, unexpanded};
use cubecl::prelude::*;
use cubecl_ir::{
    AggregateKind, Branch, ElemType, FloatKind, Instruction, MetadataKind, Operation, RangeLoop,
    SliceMetadata, Variable, VectorSize,
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
    pub fn __extract_list(&self, scope: &Scope) -> Variable {
        scope.extract_field(self.expand, self.expand.ty, SliceMetadata::LIST)
    }

    pub fn __extract_offset(&self, scope: &Scope) -> NativeExpand<usize> {
        let ty = usize::__expand_as_type(scope);
        let field = scope.extract_field(self.expand, ty, SliceMetadata::OFFSET);
        field.into()
    }

    pub fn __extract_length(&self, scope: &Scope) -> NativeExpand<usize> {
        let ty = usize::__expand_as_type(scope);
        let field = scope.extract_field(self.expand, ty, SliceMetadata::LENGTH);
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
        let item = list.value_type();

        let length = self.__extract_length(scope);
        let offset = self.__extract_offset(scope);

        let current = list.ty.vector_size();

        if vector_size == item.vector_size() {
            return self.expand.into();
        }

        let mut new_ptr = list;
        new_ptr.ty = new_ptr.ty.with_vector_size(vector_size);

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
        if T::__expand_as_type(scope) != E::__expand_as_type(scope) && !is_tf32::<E, T>(scope) {
            let elems = [
                T::__expand_as_type(scope).elem_type(),
                E::__expand_as_type(scope).elem_type(),
            ];
            let is_flex32_cast = elems.contains(&ElemType::Float(FloatKind::F32))
                && elems.contains(&ElemType::Float(FloatKind::Flex32));

            if !is_flex32_cast {
                panic!(
                    "Downcast should only be used to satisfy the Rust type system.
Expected types to be the same, got [{}, {}]",
                    elems[0], elems[1]
                )
            }
        }

        self.__expand_downcast_unchecked_method(scope)
    }

    pub fn __expand_downcast_mut_method<T: CubePrimitive>(
        &mut self,
        scope: &Scope,
    ) -> &mut SliceExpand<T> {
        if T::__expand_as_type(scope) != E::__expand_as_type(scope) && !is_tf32::<E, T>(scope) {
            let elems = [
                T::__expand_as_type(scope).elem_type(),
                E::__expand_as_type(scope).elem_type(),
            ];
            let is_flex32_cast = elems.contains(&ElemType::Float(FloatKind::F32))
                && elems.contains(&ElemType::Float(FloatKind::Flex32));

            if !is_flex32_cast {
                panic!(
                    "Downcast should only be used to satisfy the Rust type system.
Expected types to be the same, got [{}, {}]",
                    elems[0], elems[1]
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
    list: Variable,
    offset: NativeExpand<usize>,
    length: NativeExpand<usize>,
) -> SliceExpand<E> {
    let out = scope.create_aggregate(list.ty, AggregateKind::Ptr(MetadataKind::Slice));
    scope.register(Instruction::new(
        Operation::ConstructAggregate(vec![list, offset.expand, length.expand]),
        out,
    ));
    out.into()
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
        let index_ty = usize::__expand_as_type(scope);
        let len = self.__extract_length(scope).expand;

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = NativeExpand::new(i);
        let item = self
            .__expand_index_method(&child, index)
            .__expand_deref_method(&child);
        body(&child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0usize.into(),
            end: len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<'a, E: CubePrimitive> Iterable for &'a SliceExpand<E> {
    type Item = &'a E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = usize::__expand_as_type(scope);
        let len = self.__extract_length(scope).expand;

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = NativeExpand::new(i);
        let item = self.__expand_index_method(&child, index);
        body(&child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0usize.into(),
            end: len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(self, _scope: &Scope, _body: impl FnMut(&Scope, Self::Item)) {
        unimplemented!("Can't unroll slice iterator")
    }
}

impl<'a, E: CubePrimitive> Iterable for &'a mut SliceExpand<E> {
    type Item = &'a mut E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = usize::__expand_as_type(scope);
        let len = self.__extract_length(scope).expand;

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = NativeExpand::new(i);
        let item = self.__expand_index_mut_method(&child, index);
        body(&child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i,
            start: 0usize.into(),
            end: len,
            step: None,
            inclusive: false,
            scope: child,
        })));
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
        read_offset::expand::<E>(scope, self, index, None, false)
    }

    #[doc(hidden)]
    pub unsafe fn __expand_get_unchecked_mut_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut NativeExpand<E> {
        write_offset::expand::<E>(scope, self, index, None, false)
    }
}

impl<E: CubePrimitive> IndexExpand<NativeExpand<usize>> for SliceExpand<E> {
    type Output = E::ExpandType;

    fn __expand_index_method(&self, scope: &Scope, index: NativeExpand<usize>) -> &Self::Output {
        read_offset::expand::<E>(scope, self, index, None, true)
    }
}

impl<E: CubePrimitive> IndexMutExpand<NativeExpand<usize>> for SliceExpand<E> {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut Self::Output {
        write_offset::expand::<E>(scope, self, index, None, true)
    }
}

impl_slice_ranges!(SliceExpand);

impl<E: CubePrimitive> List<E> for [E] {}
impl<E: CubePrimitive> ListExpand<E> for SliceExpand<E> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope)
    }
}

impl<E: CubePrimitive> Vectorized for Box<[E]> {}
impl<E: CubePrimitive> Vectorized for [E] {}
impl<E: CubePrimitive> VectorizedExpand for SliceExpand<E> {
    fn vector_size(&self) -> VectorSize {
        self.expand.vector_size()
    }
}
impl<E: CubePrimitive> VectorizedExpand for NativeExpand<Box<[E]>> {
    fn vector_size(&self) -> VectorSize {
        self.expand.vector_size()
    }
}

mod read_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        slice: &SliceExpand<E>,
        index: NativeExpand<usize>,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a <E as cubecl::prelude::CubeType>::ExpandType {
        let list = slice.__extract_list(scope);
        let offset = slice.__extract_offset(scope);
        let index = offset.__expand_add_method(scope, index);

        expand_index_native(scope, list, index, vector_size, checked)
    }
}

mod write_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        slice: &SliceExpand<E>,
        index: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a mut E::ExpandType {
        let list = slice.__extract_list(scope);
        let offset = slice.__extract_offset(scope);
        let index = offset.__expand_add_method(scope, index);

        expand_index_mut_native(scope, list, index, vector_size, checked)
    }
}
