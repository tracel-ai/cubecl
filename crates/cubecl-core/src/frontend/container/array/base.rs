use alloc::vec::Vec;
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use cubecl_ir::{Scope, VectorSize};

use crate::frontend::{CubePrimitive, NativeExpand};
use crate::prelude::*;
use crate::{self as cubecl};
use crate::{frontend::CubeType, ir::Type, unexpanded};
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
    use cubecl_ir::AddressSpace;
    use cubecl_macros::intrinsic;

    use super::*;
    use crate::{frontend::container::slice, ir::Variable};

    #[cube]
    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create a new array of the given length.
        #[allow(unused_variables)]
        pub fn new(#[comptime] length: usize) -> Array<T> {
            intrinsic!(|scope| {
                // Allocate as a slice even though it's statically sized, so we can deref to it.
                // Unlike Rust, we can't construct fat pointers ad-hoc without access to the scope,
                // so it needs to be prepared in advance.
                let elem = T::__expand_as_type(scope);
                let ty = Type::array(elem, length, AddressSpace::Local);
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

    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create an array from data.
        #[allow(unused_variables)]
        pub fn from_data<C: CubePrimitive>(data: impl IntoIterator<Item = C>) -> Array<T> {
            unexpanded!()
        }

        /// Expand function of [`from_data`](Array::from_data).
        pub fn __expand_from_data<C: CubePrimitive>(
            scope: &Scope,
            data: ArrayData<C>,
        ) -> ArrayExpand<T> {
            let len = data.values.len();
            let buffer = scope.create_const_array(T::__expand_as_type(scope), data.values);
            let slice = slice::from_raw_parts::<T>(
                scope,
                buffer,
                0usize.into_expand(scope),
                len.into_expand(scope),
            );
            slice.expand.into()
        }
    }

    /// Type useful for the expand function of [`from_data`](Array::from_data).
    pub struct ArrayData<C> {
        values: Vec<Variable>,
        _ty: PhantomData<C>,
    }

    impl<C: CubePrimitive + Into<NativeExpand<C>>, T: IntoIterator<Item = C>> From<T> for ArrayData<C> {
        fn from(value: T) -> Self {
            let values: Vec<Variable> = value
                .into_iter()
                .map(|value| {
                    let value: NativeExpand<C> = value.into();
                    value.expand
                })
                .collect();
            ArrayData {
                values,
                _ty: PhantomData,
            }
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
        intrinsic!(|_| self.expand.ty.array_size())
    }
}

impl<C: CubePrimitive> Assign for ArrayExpand<C> {
    fn __expand_assign_method(&mut self, scope: &Scope, value: Self) {
        let value = value.__extract_list(scope);
        let arr = self.__extract_list(scope);
        assert_eq!(
            value.ty.array_size(),
            arr.ty.array_size(),
            "Can't assign differently sized arrays"
        );
        assign::expand_element(scope, value, arr);
    }

    fn init_mut(&self, _: &Scope) -> Self {
        *self
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
    fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }
}
