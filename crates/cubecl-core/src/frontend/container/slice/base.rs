use alloc::boxed::Box;

use crate::{self as cubecl, unexpanded};
use cubecl::prelude::*;
use cubecl_ir::{Branch, ElemType, FloatKind, RangeLoop, Variable, VectorSize};

#[derive(Clone, Copy)]
pub struct ReadOnly;
#[derive(Clone, Copy)]
pub struct ReadWrite;

#[derive(CubeType)]
pub enum SliceOrigin<E: CubePrimitive> {
    Tensor(Tensor<E>),
    Array(Array<E>),
    SharedMemory(SharedMemory<E>),
}

impl<E: CubePrimitive> Copy for SliceOriginExpand<E> {}
impl<E: CubePrimitive> Clone for SliceOriginExpand<E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: CubePrimitive> SliceOriginExpand<E> {
    pub fn vector_size(&self) -> VectorSize {
        match self {
            SliceOriginExpand::Tensor(t) => t.vector_size(),
            SliceOriginExpand::Array(t) => t.vector_size(),
            SliceOriginExpand::SharedMemory(t) => t.vector_size(),
        }
    }
}

pub trait SliceVisibility: Clone + Copy + Send + Sync + 'static {}

impl SliceVisibility for ReadOnly {}

impl SliceVisibility for ReadWrite {}

#[derive(Clone, Copy)]
pub struct SliceExpand<E: CubePrimitive> {
    pub(crate) origin: SliceOriginExpand<E>,
    pub(crate) offset: NativeExpand<usize>,
    pub(crate) length: NativeExpand<usize>,
    pub(crate) vector_size: Option<VectorSize>,
}

impl<E: CubePrimitive> SliceExpand<E> {
    pub fn __to_raw_parts(&self) -> (Variable, Variable) {
        let expand = match self.origin {
            SliceOriginExpand::Tensor(expand) => expand.expand,
            SliceOriginExpand::Array(expand) => expand.expand,
            SliceOriginExpand::SharedMemory(expand) => expand.expand,
        };

        (expand, self.offset.expand)
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
    fn with_vector_size_inner<N2: Size>(self, scope: &Scope) -> SliceExpand<Vector<E, N2>> {
        let vector_size = N2::__expand_value(scope);
        let (input, _) = self.__to_raw_parts();
        let item = input.ty;

        let current = input.ty.vector_size();
        let mut out = *self.__expand_downcast_unchecked_method::<Vector<E, N2>>(scope);

        if vector_size == item.vector_size() {
            return out;
        }

        if current < vector_size {
            let ratio = vector_size / current;
            let length = self.length.__expand_div_method(scope, ratio.into());
            let offset = self.offset.__expand_div_method(scope, ratio.into());
            out.length = length;
            out.offset = offset;
        } else {
            let ratio = current / vector_size;
            let length = self.length.__expand_mul_method(scope, ratio.into());
            let offset = self.offset.__expand_mul_method(scope, ratio.into());
            out.length = length;
            out.offset = offset;
        }

        out.vector_size = Some(vector_size);
        out
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

    fn as_ptr(&self) -> &E {
        unexpanded!()
    }

    fn as_ptr_mut(&mut self) -> &mut E {
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

    fn __expand_as_ptr<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer NativeExpand<E>;

    fn __expand_as_ptr_mut<'infer>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut NativeExpand<E>;

    #[allow(clippy::mut_from_ref)]
    #[doc(hidden)]
    unsafe fn __expand_as_mut_unchecked<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer mut SliceExpand<E>;

    #[doc(hidden)]
    unsafe fn __expand_as_boxed_unchecked(scope: &Scope, this: &SliceExpand<E>) -> SliceExpand<E>;
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

    fn __expand_as_ptr<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer NativeExpand<E> {
        this.__expand_as_ptr_method(scope)
    }

    fn __expand_as_ptr_mut<'infer>(
        scope: &Scope,
        this: &'infer mut SliceExpand<E>,
    ) -> &'infer mut NativeExpand<E> {
        this.__expand_as_ptr_mut_method(scope)
    }

    unsafe fn __expand_as_mut_unchecked<'infer>(
        scope: &Scope,
        this: &'infer SliceExpand<E>,
    ) -> &'infer mut SliceExpand<E> {
        this.__expand_as_mut_unchecked_method(scope)
    }

    unsafe fn __expand_as_boxed_unchecked(scope: &Scope, this: &SliceExpand<E>) -> SliceExpand<E> {
        this.__expand_as_boxed_unchecked_method(scope)
    }
}

impl<E: CubePrimitive> SliceExpand<E> {
    pub fn __expand_as_vectorized_method(
        &self,
        scope: &Scope,
    ) -> &SliceExpand<Vector<E::Scalar, E::Size>> {
        let slice = SliceExpand::<Vector<E::Scalar, E::Size>> {
            origin: self.origin.cast_unchecked(),
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }

    pub fn __expand_as_vectorized_mut_method(
        &mut self,
        scope: &Scope,
    ) -> &mut SliceExpand<Vector<E::Scalar, E::Size>> {
        let slice = SliceExpand::<Vector<E::Scalar, E::Size>> {
            origin: self.origin.cast_unchecked(),
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
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
        scope: &Scope,
    ) -> &SliceExpand<T> {
        let slice = SliceExpand::<T> {
            origin: self.origin.cast_unchecked(),
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }

    #[doc(hidden)]
    pub fn __expand_downcast_mut_unchecked_method<T: CubePrimitive>(
        &mut self,
        scope: &Scope,
    ) -> &mut SliceExpand<T> {
        let slice = SliceExpand::<T> {
            origin: self.origin.cast_unchecked(),
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        };
        scope.create_kernel_ref(slice)
    }

    pub fn __expand_as_ptr_method(&self, scope: &Scope) -> &NativeExpand<E> {
        as_ptr::expand(scope, &self.origin, self.offset, self.vector_size, true)
    }

    pub fn __expand_as_ptr_mut_method(&mut self, scope: &Scope) -> &mut NativeExpand<E> {
        as_ptr_mut::expand(scope, &mut self.origin, self.offset, self.vector_size, true)
    }

    pub fn __expand_as_mut_unchecked_method(&self, scope: &Scope) -> &mut SliceExpand<E> {
        scope.create_kernel_ref(*self)
    }

    pub fn __expand_as_boxed_unchecked_method(&self, _: &Scope) -> SliceExpand<E> {
        *self
    }
}

impl<E: CubePrimitive> SliceOriginExpand<E> {
    fn cast_unchecked<T: CubePrimitive>(self) -> SliceOriginExpand<T> {
        match self {
            SliceOriginExpand::Tensor(expand) => {
                SliceOriginExpand::<T>::Tensor(expand.expand.into())
            }
            SliceOriginExpand::Array(expand) => SliceOriginExpand::<T>::Array(expand.expand.into()),
            SliceOriginExpand::SharedMemory(expand) => {
                SliceOriginExpand::<T>::SharedMemory(expand.expand.into())
            }
        }
    }
}

impl<E: CubePrimitive> SliceExpand<E> {
    pub fn new(
        scope: &Scope,
        origin: SliceOriginExpand<E>,
        start: NativeExpand<usize>,
        end: NativeExpand<usize>,
    ) -> SliceExpand<E> {
        let length = cubecl::frontend::sub::expand(scope, end, start);

        SliceExpand::<E> {
            origin,
            offset: start,
            length,
            vector_size: None,
        }
    }

    /// Get the length of the slice.
    pub fn __expand_len_method(&self, _: &Scope) -> NativeExpand<usize> {
        self.length
    }
    /// Returns true if the slice is empty.
    pub fn is_empty(&self, scope: &Scope) -> NativeExpand<bool> {
        self.length
            .__expand_eq_method(scope, &0usize.into_expand(scope))
    }
}

impl<E: CubePrimitive> CubeType for [E] {
    type ExpandType = SliceExpand<E>;
}

impl<E: CubePrimitive> CubeType for Box<[E]> {
    type ExpandType = SliceExpand<E>;
}

impl<'a, E: CubePrimitive> CubeType for &'a [E] {
    type ExpandType = &'a SliceExpand<E>;
}

impl<'a, E: CubePrimitive> CubeType for &'a mut [E] {
    type ExpandType = &'a mut SliceExpand<E>;
}

macro_rules! impl_expand_traits {
    ($generic: ident, $ty: ty) => {
        impl<$generic: CubePrimitive> AsRefExpand for $ty {
            fn __expand_as_ref_method(&self, _: &Scope) -> &Self {
                self
            }
        }
        impl<$generic: CubePrimitive> AsMutExpand for $ty {
            fn __expand_as_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }

        impl<$generic: CubePrimitive> IntoExpand for $ty {
            type Expand = Self;

            fn into_expand(self, _scope: &Scope) -> Self::Expand {
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
impl_expand_traits!(E, &SliceExpand<E>);
impl_expand_traits!(E, &mut SliceExpand<E>);

impl<E: CubePrimitive> ExpandTypeClone for SliceExpand<E> {
    fn clone_unchecked(&self) -> Self {
        SliceExpand {
            origin: self.origin.clone_unchecked(),
            offset: self.offset,
            length: self.length,
            vector_size: self.vector_size,
        }
    }
}

impl<E: CubePrimitive> ExpandTypeClone for &SliceExpand<E> {
    fn clone_unchecked(&self) -> Self {
        self
    }
}

impl<E: CubePrimitive> ExpandTypeClone for &mut SliceExpand<E> {
    #[allow(mutable_transmutes)]
    fn clone_unchecked(&self) -> Self {
        unsafe { core::mem::transmute(self) }
    }
}

impl<E: CubePrimitive> CubeDebug for SliceExpand<E> {}

impl<E: CubePrimitive> SizedContainer for [E] {
    type Item = E;
}

impl<E: CubePrimitive> Iterable for SliceExpand<E> {
    type Item = E::ExpandType;

    fn expand(self, scope: &Scope, mut body: impl FnMut(&Scope, Self::Item)) {
        let index_ty = u32::__expand_as_type(scope);
        let len: Variable = self.length.into();

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.into();
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
        let index_ty = u32::__expand_as_type(scope);
        let len: Variable = self.length.into();

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.into();
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
        let index_ty = u32::__expand_as_type(scope);
        let len: Variable = self.length.into();

        let child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.into();
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

impl<E: CubePrimitive> CubeIndex for Box<[E]> {
    type Output = E;
    type Idx = usize;
}

impl<E: CubePrimitive> CubeIndex for [E] {
    type Output = E;
    type Idx = usize;
}

impl<E: CubePrimitive> CubeIndexExpand for SliceExpand<E> {
    type Output = E::ExpandType;
    type Idx = NativeExpand<usize>;

    fn __expand_index_method(&self, scope: &Scope, index: NativeExpand<usize>) -> &Self::Output {
        self.__expand_read_method(scope, index)
    }
    fn __expand_index_unchecked_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &Self::Output {
        self.__expand_read_unchecked_method(scope, index)
    }
}

impl<E: CubePrimitive> List<E> for Box<[E]> {}
impl<E: CubePrimitive> List<E> for [E] {}
impl<E: CubePrimitive> ListExpand<E> for SliceExpand<E> {
    fn __expand_read_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &<E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            &self.origin,
            self.offset,
            index,
            self.vector_size,
            true,
        )
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &<E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            &self.origin,
            self.offset,
            index,
            self.vector_size,
            false,
        )
    }

    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope)
    }
}

impl<T: CubePrimitive> DerefExpand for SliceExpand<T> {
    type Target = Self;

    fn __expand_deref_method(&self, _: &Scope) -> Self::Target {
        *self
    }
}

impl<E: CubePrimitive> Vectorized for Box<[E]> {}
impl<E: CubePrimitive> Vectorized for [E] {}
impl<E: CubePrimitive> VectorizedExpand for SliceExpand<E> {
    fn vector_size(&self) -> VectorSize {
        self.vector_size
            .unwrap_or_else(|| self.origin.vector_size())
    }
}

impl<E: CubePrimitive> CubeIndexMut for Box<[E]> {}
impl<E: CubePrimitive> CubeIndexMut for [E] {}
impl<E: CubePrimitive> CubeIndexMutExpand for SliceExpand<E> {
    fn __expand_index_mut_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut Self::Output {
        self.__expand_write_method(scope, index)
    }
}

impl<E: CubePrimitive> ListMut<E> for Box<[E]> {}
impl<E: CubePrimitive> ListMut<E> for [E] {}
impl<E: CubePrimitive> ListMutExpand<E> for SliceExpand<E> {
    fn __expand_write_method(
        &self,
        scope: &Scope,
        index: NativeExpand<usize>,
    ) -> &mut NativeExpand<E> {
        let mut origin = self.origin;
        let reference =
            write_offset::expand::<E>(scope, &mut origin, self.offset, index, self.vector_size);
        // Safety: Cloning origin only clones the reference, so this is safe
        unsafe { core::mem::transmute(reference) }
    }
}

mod as_ptr {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a <E as cubecl::prelude::CubeType>::ExpandType {
        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_native(scope, expand, offset, vector_size, checked)
            }
        }
    }
}

mod as_ptr_mut {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a mut SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a mut <E as cubecl::prelude::CubeType>::ExpandType {
        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_mut_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_mut_native(scope, expand, offset, vector_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_mut_native(scope, expand, offset, vector_size, checked)
            }
        }
    }
}

mod read_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        index: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
        checked: bool,
    ) -> &'a <E as cubecl::prelude::CubeType>::ExpandType {
        let index = offset.__expand_add_method(scope, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_native(scope, expand, index, vector_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_native(scope, expand, index, vector_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_native(scope, expand, index, vector_size, checked)
            }
        }
    }
}

mod write_offset {
    use super::*;

    pub fn expand<'a, E: CubePrimitive>(
        scope: &Scope,
        origin: &'a mut SliceOriginExpand<E>,
        offset: <usize as CubeType>::ExpandType,
        index: <usize as CubeType>::ExpandType,
        vector_size: Option<VectorSize>,
    ) -> &'a mut E::ExpandType {
        let index = offset.__expand_add_method(scope, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_mut_native(scope, expand, index, vector_size, true)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_mut_native(scope, expand, index, vector_size, false)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_mut_native(scope, expand, index, vector_size, false)
            }
        }
    }
}
