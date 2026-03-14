use alloc::vec::Vec;
use core::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use cubecl_ir::{ManagedVariable, Scope, VectorSize};

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

type ArrayExpand<E> = NativeExpand<Array<E>>;

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
                    *value.expand
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
        pub fn to_vectorized<N: Size>(self) -> T {
            let factor = N::value();
            intrinsic!(|scope| {
                let var = self.expand.clone();
                let item = Type::new(var.storage_type()).with_vector_size(factor);

                let new_var = if factor == 1 {
                    let new_var = scope.create_local(item);
                    let element =
                        index::expand(scope, self.clone(), NativeExpand::from_lit(scope, 0));
                    assign::expand_no_check::<T>(scope, element, new_var.clone().into());
                    new_var
                } else {
                    let new_var = scope.create_local_mut(item);
                    for i in 0..factor {
                        let expand: Self = self.expand.clone().into();
                        let element =
                            index::expand(scope, expand, NativeExpand::from_lit(scope, i));
                        index_assign::expand::<NativeExpand<Array<T>>, T>(
                            scope,
                            new_var.clone().into(),
                            NativeExpand::from_lit(scope, i),
                            element,
                        );
                    }
                    new_var
                };
                new_var.into()
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
            intrinsic!(|scope| {
                ManagedVariable::Plain(expand_length_native(scope, *self.expand)).into()
            })
        }

        /// Obtain the array buffer length
        pub fn buffer_len(&self) -> usize {
            intrinsic!(|scope| {
                let out = scope.create_local(usize::as_type(scope));
                scope.register(Instruction::new(
                    Metadata::BufferLength {
                        var: self.expand.into(),
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
    use cubecl_ir::{IndexAssignOperator, IndexOperator, Operator};

    use crate::ir::Instruction;

    use super::*;

    #[cube]
    impl<E: CubePrimitive> Array<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&self, i: usize) -> &E {
            intrinsic!(|scope| {
                let out = scope.create_local(self.expand.ty);
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: *self.expand,
                        index: i.expand.consume(),
                        vector_size: 0,
                        unroll_factor: 1,
                    }),
                    *out,
                ));
                out.into()
            })
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_assign_unchecked(&mut self, i: usize, value: E) {
            intrinsic!(|scope| {
                scope.register(Instruction::new(
                    Operator::UncheckedIndexAssign(IndexAssignOperator {
                        index: i.expand.consume(),
                        value: value.expand.consume(),
                        vector_size: 0,
                        unroll_factor: 1,
                    }),
                    *self.expand,
                ));
            })
        }
    }
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = NativeExpand<Array<C>>;
}

impl<C: CubeType> CubeType for &Array<C> {
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

impl<T: CubeType> Iterator for &Array<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

impl<T: CubePrimitive> List<T> for Array<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: NativeExpand<Array<T>>,
        idx: NativeExpand<usize>,
    ) -> NativeExpand<T> {
        index::expand(scope, this, idx)
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

impl<T: CubePrimitive> ListExpand<T> for NativeExpand<Array<T>> {
    fn __expand_read_method(&self, scope: &mut Scope, idx: NativeExpand<usize>) -> NativeExpand<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
    ) -> NativeExpand<T> {
        index_unchecked::expand(scope, self.clone(), idx)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> NativeExpand<usize> {
        Self::__expand_len(scope, self.clone())
    }
}

impl<T: CubePrimitive> Vectorized for Array<T> {}
impl<T: CubePrimitive> VectorizedExpand for NativeExpand<Array<T>> {
    fn vector_size(&self) -> VectorSize {
        self.expand.ty.vector_size()
    }
}

impl<T: CubePrimitive> ListMut<T> for Array<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: NativeExpand<Array<T>>,
        idx: NativeExpand<usize>,
        value: NativeExpand<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for NativeExpand<Array<T>> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        idx: NativeExpand<usize>,
        value: NativeExpand<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
