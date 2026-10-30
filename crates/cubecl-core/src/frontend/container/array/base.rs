use core::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use alloc::vec::Vec;
use cubecl_ir::{
    Scope, VectorSize,
    interfaces::MaybeVectorizedType,
    pliron::{
        r#type::{Typed, TypedHandle},
        value::Value,
    },
    prelude::TypeHandle,
    read_value,
    types::{ArrayType, PointerType, aggregate::SliceType},
};

use crate::frontend::{
    CubePrimitive, NativeExpand,
    container::iter::{SizedContainer, SizedContainerExpand},
};
use crate::prelude::*;
use crate::{self as cubecl};
use crate::{frontend::CubeType, unexpanded};
use cubecl_macros::{cube, intrinsic};

/// A contiguous array of elements.
#[derive(Clone, Copy)]
pub struct Array<E> {
    _buffer: [E; 1],
}

type ArrayExpand<E> = NativeExpand<Array<E>>;

impl<E> AsMutExpand for ArrayExpand<E> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

impl<E, const N: usize> AsMutExpand for NativeExpand<[E; N]> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

/// Module that contains the implementation details of the new function.
mod new {
    use cubecl_ir::{
        attributes::ZeroAttr, dialect::vector::CompositeConstructOp, types::ArrayType,
    };
    use cubecl_macros::intrinsic;

    use super::*;
    use crate::frontend::container::slice;

    #[cube]
    impl<T: CubePrimitive> Array<T> {
        /// Create a new array of the given length.
        pub fn new(#[comptime] length: usize) -> Self {
            intrinsic!(|scope| {
                // Allocate as a slice even though it's statically sized, so we can deref to it.
                // Unlike Rust, we can't construct fat pointers ad-hoc without access to the scope,
                // so it needs to be prepared in advance.
                let elem = T::__expand_as_type(scope);
                let ty = ArrayType::get(scope.ctx(), elem, length);
                let null = ZeroAttr::new(ty);
                let buffer = scope.create_local_mut(ty, Some(null.into()));
                let slice = slice::from_raw_parts::<T>(
                    scope,
                    buffer,
                    0usize.into_expand(scope),
                    length.into_expand(scope),
                );
                slice.expand.into()
            })
        }
    }

    impl<T: CubePrimitive> ArrayExpand<T> {
        pub fn __expand_clone_method(&self, scope: &Scope) -> Self {
            Self::into_mut(*self, scope)
        }
    }

    #[doc(hidden)]
    impl<T: CubePrimitive, const N: usize> NativeExpand<[T; N]> {
        pub fn __expand_zeroed(scope: &Scope) -> Self {
            let elem = T::__expand_as_type(scope);
            let ty = ArrayType::get(scope.ctx(), elem, N);
            let null = ZeroAttr::new(ty);
            let buffer = scope.create_local_mut(ty, Some(null.into()));
            let slice = slice::from_raw_parts::<T>(
                scope,
                buffer,
                0usize.into_expand(scope),
                N.into_expand(scope),
            );
            slice.expand.into()
        }

        pub fn __expand_construct(scope: &Scope, values: [NativeExpand<T>; N]) -> Self {
            let length = values.len();
            let values = values.into_iter().map(|it| it.read_value(scope)).collect();
            let elem = T::__expand_as_type(scope);
            let ty = ArrayType::get(scope.ctx(), elem, length).to_handle();
            let value =
                scope.register_with_result(&CompositeConstructOp::new(scope.ctx_mut(), ty, values));
            let uninit = Self::__expand_zeroed(scope);
            assign::expand_element(scope, value.into(), uninit.__extract_list(scope).into());
            uninit
        }

        pub fn __expand_clone_method(&self, scope: &Scope) -> Self {
            Self::into_mut(*self, scope)
        }
    }
}

/// Module that contains the implementation details of the `vector_size` function.
mod vector {
    use cubecl_ir::{interfaces::TypedExt, read_value};

    use super::*;

    impl<P: CubePrimitive> Array<P> {
        /// Get the size of each vector contained in the array.
        ///
        /// Same as the following:
        ///
        /// ```rust, ignore
        /// let size = tensor[0].vector_size();
        /// ```
        pub fn vector_size(&self) -> VectorSize {
            P::vector_size()
        }

        // Expand function of [size](Tensor::vector_size).
        pub fn __expand_vector_size(
            expand: <Self as CubeType>::ExpandType,
            scope: &Scope,
        ) -> VectorSize {
            expand.__expand_vector_size_method(scope)
        }
    }

    #[cube]
    impl<P: CubePrimitive> Array<P> {
        pub fn into_vector<N: Size>(self) -> Vector<P::Scalar, N> {
            intrinsic!(|scope| {
                let arr = read_value(scope, self.__extract_list(scope));
                let vec_ty = Vector::<P::Scalar, N>::__expand_as_type(scope);
                reinterpret_value(scope, arr, vec_ty).into()
            })
        }

        pub fn from_vector<S: Scalar, N: Size>(vector: Vector<S, N>) -> Array<P> {
            intrinsic!(|scope| {
                let vec = vector.read_value(scope);
                let vec_p = P::__expand_vector_size(scope);
                let len = vec.vector_size(scope.ctx()) / vec_p;
                let arr_ty =
                    ArrayType::get(scope.ctx(), P::__expand_as_type(scope), len).to_handle();
                reinterpret_value(scope, vec, arr_ty).into()
            })
        }
    }
}

#[cube]
impl<E: CubePrimitive> Array<E> {
    /// Obtain the array length
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            let ty = inner_array_ty(scope, self.value(scope));
            ty.deref(scope.ctx()).length
        })
    }
}

impl<T: CubePrimitive, const N: usize> NativeExpand<[T; N]> {
    pub fn __expand_len_method(&self, scope: &cubecl::prelude::Scope) -> usize {
        let ty = inner_array_ty(scope, self.value(scope));
        ty.deref(scope.ctx()).length
    }
}

impl<C: CubePrimitive> Assign for ArrayExpand<C> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
        let value = value.read_value(scope);
        let arr = self.__extract_list(scope);
        assign::expand_element(scope, value.into(), arr.into());
    }
}

impl<C: CubePrimitive, const N: usize> Assign for NativeExpand<[C; N]> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
        let value = value.read_value(scope);
        let arr = self.__extract_list(scope);
        assign::expand_element(scope, value.into(), arr.into());
    }
}

impl<C: CubePrimitive> RuntimeAssign for ArrayExpand<C> {
    fn init_mut(&self, scope: &Scope) -> Self::Expand {
        let ty = inner_array_ty(scope, self.value(scope));
        let length = ty.deref(scope.ctx()).length;
        Array::__expand_new(scope, length)
    }
}

impl<C: CubePrimitive, const N: usize> RuntimeAssign for NativeExpand<[C; N]> {
    fn init_mut(&self, scope: &Scope) -> Self::Expand {
        Self::__expand_zeroed(scope)
    }
}
impl<C: CubePrimitive, const N: usize> ConstRuntimeAssign for NativeExpand<[C; N]> {
    fn const_init_mut(scope: &Scope) -> Self::Expand {
        Self::__expand_zeroed(scope)
    }
}

impl<C: CubePrimitive> CubeType for Array<C> {
    type ExpandType = NativeExpand<Array<C>>;
}

impl<C: CubePrimitive, const N: usize> CubeType for [C; N] {
    type ExpandType = NativeExpand<[C; N]>;
}

impl<T: CubePrimitive, const N: usize> ConstantTyped for [T; N] {
    fn __expand_as_type(scope: &Scope) -> TypeHandle {
        let inner = T::__expand_as_type(scope);
        ArrayType::get(scope.ctx(), inner, N).to_handle()
    }
}

impl<T: CubePrimitive> ReadValue for NativeExpand<Array<T>> {
    fn read_value(&self, scope: &Scope) -> Value {
        read_value(scope, self.__extract_list(scope))
    }
}

impl<T: CubePrimitive, const N: usize> ReadValue for NativeExpand<[T; N]> {
    fn read_value(&self, scope: &Scope) -> Value {
        read_value(scope, self.__extract_list(scope))
    }
}

impl<C: CubePrimitive> IntoMut for ArrayExpand<C> {
    fn into_mut(self, scope: &Scope) -> Self {
        let mut out = Array::__expand_new(scope, self.__expand_len_method(scope));
        out.__expand_assign_method(scope, self);
        out
    }
}

impl<C: CubePrimitive, const N: usize> IntoMut for NativeExpand<[C; N]> {
    fn into_mut(self, scope: &Scope) -> Self {
        let mut uninit = Self::__expand_zeroed(scope);
        uninit.__expand_assign_method(scope, self);
        uninit
    }
}

impl<T: CubePrimitive> SizedContainer<usize> for Array<T> {
    fn len(&self) -> usize {
        unexpanded!()
    }
}

impl<T: CubePrimitive> SizedContainerExpand<usize> for ArrayExpand<T> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope).into_expand(scope)
    }
}

impl<T: CubePrimitive, const N: usize> SizedContainer<usize> for [T; N] {
    fn len(&self) -> usize {
        N
    }
}

impl<T: CubePrimitive, const N: usize> SizedContainerExpand<usize> for NativeExpand<[T; N]> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        N.into_expand(scope)
    }
}

impl<T: CubeType> Iterator for Array<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

impl<T: CubePrimitive> Deref for Array<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> DerefMut for Array<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> Deref for ArrayExpand<T> {
    type Target = SliceExpand<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.as_type_ref_unchecked() }
    }
}

impl<T: CubePrimitive> DerefMut for ArrayExpand<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_type_mut_unchecked() }
    }
}

impl<T: CubePrimitive, const N: usize> Deref for NativeExpand<[T; N]> {
    type Target = SliceExpand<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { self.as_type_ref_unchecked() }
    }
}

impl<T: CubePrimitive, const N: usize> DerefMut for NativeExpand<[T; N]> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_type_mut_unchecked() }
    }
}

impl<T: CubePrimitive> List<T> for Array<T> {}
impl<T: CubePrimitive> ListExpand<T> for ArrayExpand<T> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        Array::<T>::__expand_len(scope, self).into_expand(scope)
    }
}

impl<T: CubePrimitive> Vectorized for Array<T> {}
impl<T: CubePrimitive> VectorizedExpand for ArrayExpand<T> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        let ty = inner_array_ty(scope, self.value(scope));
        ty.deref(scope.ctx()).vector_size(scope.ctx())
    }
}

impl<T: CubePrimitive, const N: usize> List<T> for [T; N] {}
impl<T: CubePrimitive, const N: usize> ListExpand<T> for NativeExpand<[T; N]> {
    fn __expand_len_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.__expand_len_method(scope).into_expand(scope)
    }
}

impl<T: CubePrimitive, const N: usize> Vectorized for [T; N] {}
impl<T: CubePrimitive, const N: usize> VectorizedExpand for NativeExpand<[T; N]> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        let ty = inner_array_ty(scope, self.value(scope));
        ty.deref(scope.ctx()).vector_size(scope.ctx())
    }
}

pub(crate) fn inner_array_ty(scope: &Scope, value: Value) -> TypedHandle<ArrayType> {
    let ctx = scope.ctx();
    let ty = value.get_type(ctx).deref(ctx);
    let SliceType { base_ty, .. } = *ty.downcast_ref().unwrap();
    let base_ty = base_ty.deref(ctx);
    let PointerType { inner, .. } = base_ty.downcast_ref().unwrap();
    TypedHandle::from_handle(*inner, ctx).unwrap()
}

impl<T: LaunchArg + CubePrimitive + Debug + 'static, const N: usize> LaunchArg for [T; N] {
    type RuntimeArg = [T::RuntimeArg; N];
    type CompilationArg = [T::CompilationArg; N];

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        arg.into_iter()
            .map(|arg| T::register(arg, launcher))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<[T; N]> {
        let values = arg
            .iter()
            .map(|value| T::expand(value, builder))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        NativeExpand::<[T; N]>::__expand_construct(builder, values)
    }
}
