use core::ops::{Deref, DerefMut};

use cubecl_ir::{
    Scope, VectorSize,
    interfaces::MaybeVectorizedType,
    pliron::{
        r#type::{TypePtr, Typed},
        value::Value,
    },
    types::{ArrayType, PointerType, aggregate::PtrAggregateType},
};

use crate::frontend::{CubePrimitive, NativeExpand};
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

/// Module that contains the implementation details of the new function.
mod new {
    use cubecl_ir::types::ArrayType;
    use cubecl_macros::intrinsic;

    use super::*;
    use crate::frontend::container::slice;

    #[cube]
    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create a new array of the given length.
        pub fn new(#[comptime] length: usize) -> Self {
            intrinsic!(|scope| {
                // Allocate as a slice even though it's statically sized, so we can deref to it.
                // Unlike Rust, we can't construct fat pointers ad-hoc without access to the scope,
                // so it needs to be prepared in advance.
                let elem = T::__expand_as_type(scope);
                let ty = ArrayType::get(&mut scope.ctx_mut(), elem, length);
                let buffer = scope.create_local_mut(ty);
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
}

/// Module that contains the implementation details of the `vector_size` function.
mod vector {
    use super::*;

    impl<P: CubePrimitive> Array<P> {
        /// Get the size of each vector contained in the tensor.
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
}

#[cube]
impl<E: CubePrimitive> Array<E> {
    /// Obtain the array length
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            let ty = inner_array_ty(scope, self.value(scope));
            ty.deref(&scope.ctx()).length
        })
    }
}

impl<C: CubePrimitive> Assign for ArrayExpand<C> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
        let value = value.__extract_list(scope);
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

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = NativeExpand<Array<C>>;
}

impl<C: CubeType> IntoMut for ArrayExpand<C> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
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

pub(crate) fn inner_array_ty(scope: &Scope, value: Value) -> TypePtr<ArrayType> {
    let ctx = scope.ctx();
    let ty = value.get_type(ctx).deref(ctx);
    let PtrAggregateType { base_ty, .. } = *ty.downcast_ref().unwrap();
    let base_ty = base_ty.deref(ctx);
    let PointerType { inner, .. } = base_ty.downcast_ref().unwrap();
    TypePtr::from_ptr(*inner, ctx).unwrap()
}
