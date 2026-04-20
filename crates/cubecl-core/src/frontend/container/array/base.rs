use alloc::vec::Vec;
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use cubecl_ir::{Scope, VectorSize};

use crate::frontend::{CubePrimitive, NativeExpand};
use crate::prelude::*;
use crate::{self as cubecl};
use crate::{
    frontend::CubeType,
    ir::{Metadata, Type},
    unexpanded,
};
use cubecl_macros::{cube, intrinsic};

/// A contiguous array of elements.
pub struct Array<E> {
    _val: PhantomData<E>,
}

impl<E> Copy for Array<E> {}
impl<E> Clone for Array<E> {
    fn clone(&self) -> Self {
        *self
    }
}

type ArrayExpand<E> = NativeExpand<Array<E>>;

impl<E> ExpandAsRef for ArrayExpand<E> {
    fn __expand_as_ref_method<'a>(&'a self, _: &mut Scope) -> &'a Self {
        self
    }

    fn __expand_as_mut_method<'a>(&'a mut self, _: &mut Scope) -> &'a mut Self {
        self
    }
}

/// Module that contains the implementation details of the new function.
mod new {

    use cubecl_macros::intrinsic;

    use super::*;
    use crate::ir::Variable;

    #[cube]
    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create a new array of the given length.
        #[allow(unused_variables)]
        pub fn new(#[comptime] length: usize) -> Self {
            intrinsic!(|scope| {
                let elem = T::as_type(scope);
                scope.create_local_array(elem, length).into()
            })
        }
    }

    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create an array from data.
        #[allow(unused_variables)]
        pub fn from_data<C: CubePrimitive>(data: impl IntoIterator<Item = C>) -> Self {
            intrinsic!(|scope| {
                scope
                    .create_const_array(Type::new(T::as_type(scope)), data.values)
                    .into()
            })
        }

        /// Expand function of [`from_data`](Array::from_data).
        pub fn __expand_from_data<C: CubePrimitive>(
            scope: &mut Scope,
            data: ArrayData<C>,
        ) -> <Self as CubeType>::ExpandType {
            let var = scope.create_const_array(T::as_type(scope), data.values);
            NativeExpand::new(var)
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
            scope: &mut Scope,
        ) -> VectorSize {
            expand.__expand_vector_size_method(scope)
        }
    }
}

/// Module that contains the implementation details of vectorization functions.
mod vectorization {
    use super::*;

    #[cube]
    impl<T: CubePrimitive + Clone> Array<T> {
        #[allow(unused_variables)]
        pub fn as_vectorized<N: Size>(&self) -> Vector<T::Scalar, N> {
            let factor = N::value();
            intrinsic!(|scope| {
                let var = self.expand.clone();
                let item = Type::new(var.storage_type()).with_vector_size(factor);

                if factor == 1 {
                    let new_var = scope.create_local(item);
                    let element = self
                        .__expand_index_method(scope, NativeExpand::from_lit(scope, 0))
                        .__expand_deref_method(scope);
                    assign::expand_no_check::<T>(scope, element, new_var.clone().into());
                    new_var.into()
                } else {
                    let mut new_var: NativeExpand<Vector<T::Scalar, N>> =
                        scope.create_local_mut(item).into();
                    for i in 0..factor {
                        let idx = NativeExpand::from_lit(scope, i);
                        let element = self
                            .__expand_index_method(scope, idx)
                            .__expand_deref_method(scope);

                        new_var.__expand_insert_method(scope, idx, element.expand.into());
                    }
                    new_var
                }
            })
        }
    }
}

/// Module that contains the implementation details of the metadata functions.
mod metadata {
    use crate::{ir::Instruction, prelude::expand_length_native};

    use super::*;

    #[cube]
    impl<E: CubeType> Array<E> {
        /// Obtain the array length
        #[allow(clippy::len_without_is_empty)]
        pub fn len(&self) -> usize {
            intrinsic!(|scope| { expand_length_native(scope, self.expand).into() })
        }

        /// Obtain the array buffer length
        pub fn buffer_len(&self) -> usize {
            intrinsic!(|scope| {
                let out = scope.create_local(usize::as_type(scope));
                scope.register(Instruction::new(
                    Metadata::BufferLength {
                        var: self.expand.clone().into(),
                    },
                    out.clone().into(),
                ));
                out.into()
            })
        }
    }
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::{IndexMutOperator, IndexOperator, Operator};

    use crate::ir::Instruction;

    use super::*;

    #[cube]
    impl<'a, E: CubePrimitive> Array<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&'a self, i: usize) -> &'a E {
            intrinsic!(|scope| {
                let ty = self.expand.ty;
                let class = self.expand.pointer_class();
                let out = scope.create_local(Type::pointer(ty, class));
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: self.expand,
                        index: i.expand,
                        vector_size: 0,
                        unroll_factor: 1,
                    }),
                    out,
                ));
                scope.create_kernel_ref(out.into())
            })
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_mut_unchecked(&'a mut self, i: usize) -> &'a mut E {
            intrinsic!(|scope| {
                let ty = self.expand.ty;
                let class = self.expand.pointer_class();
                let out = scope.create_local(Type::pointer(ty, class));
                scope.register(Instruction::new(
                    Operator::UncheckedIndexMut(IndexMutOperator {
                        list: self.expand,
                        index: i.expand,
                        vector_size: 0,
                        unroll_factor: 1,
                    }),
                    out,
                ));
                scope.create_kernel_ref(out.into())
            })
        }
    }
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = NativeExpand<Array<C>>;
}

impl<C: CubeType> IntoMut for NativeExpand<Array<C>> {
    fn into_mut(self, _scope: &mut crate::ir::Scope) -> Self {
        // The type can't be deeply cloned/copied.
        self
    }
}

impl<T: CubePrimitive> SizedContainer for Array<T> {
    type Item = T;
}

impl<T: CubeType> Iterator for Array<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

impl<'a, T: CubePrimitive> List<'a, T> for Array<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: &'a NativeExpand<Array<T>>,
        idx: NativeExpand<usize>,
    ) -> &'a NativeExpand<T> {
        this.__expand_index_method(scope, idx)
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

impl<T: CubePrimitive> ExpandDeref for ArrayExpand<T> {
    type Target = ArrayExpand<T>;

    fn __expand_deref_method(&self, _: &mut Scope) -> Self::Target {
        *self
    }
}

impl<'a, T: CubePrimitive> ListExpand<'a, T> for ArrayExpand<T> {
    fn __expand_read_method(
        &'a self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> &'a NativeExpand<T> {
        self.__expand_index_method(scope, idx)
    }
    fn __expand_read_unchecked_method(
        &'a self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> &'a NativeExpand<T> {
        self.__expand_index_unchecked_method(scope, idx)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> NativeExpand<usize> {
        Array::<T>::__expand_len(scope, self)
    }
}

impl<T: CubePrimitive> Vectorized for Array<T> {}
impl<T: CubePrimitive> VectorizedExpand for ArrayExpand<T> {
    fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }
}

impl<'a, T: CubePrimitive> ListMut<'a, T> for Array<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: &'a ArrayExpand<T>,
        idx: NativeExpand<usize>,
    ) -> &'a mut NativeExpand<T> {
        let mut this = *this;
        let reference = this.__expand_index_mut_method(scope, idx);
        // Extend lifetime because we know the array is actually 'a
        unsafe { core::mem::transmute(reference) }
    }
}

impl<'a, T: CubePrimitive> ListMutExpand<'a, T> for ArrayExpand<T> {
    fn __expand_write_method(
        &'a self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> &'a mut NativeExpand<T> {
        let mut this = *self;
        let reference = this.__expand_index_mut_method(scope, idx);
        // Extend lifetime because we know the array is actually 'a
        unsafe { core::mem::transmute(reference) }
    }
}
