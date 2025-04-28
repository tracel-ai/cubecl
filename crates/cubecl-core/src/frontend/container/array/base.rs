use std::{marker::PhantomData, num::NonZero};

use cubecl_ir::{ExpandElement, Scope};

use crate as cubecl;
use crate::frontend::{CubePrimitive, ExpandElementBaseInit, ExpandElementTyped};
use crate::prelude::{List, ListExpand, ListMut, ListMutExpand, SizedContainer, index_unchecked};
use crate::prelude::{assign, index, index_assign};
use crate::{
    frontend::CubeType,
    ir::{Item, Metadata},
    unexpanded,
};
use cubecl_macros::{cube, intrinsic};

/// A contiguous array of elements.
pub struct Array<E> {
    _val: PhantomData<E>,
}

type ArrayExpand<E> = ExpandElementTyped<Array<E>>;

/// Module that contains the implementation details of the new function.
mod new {

    use cubecl_macros::intrinsic;

    use super::*;
    use crate::ir::Variable;

    #[cube]
    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create a new array of the given length.
        #[allow(unused_variables)]
        pub fn new(length: u32) -> Self {
            intrinsic!(|scope| {
                let size = length
                    .constant()
                    .expect("Array needs constant initialization value")
                    .as_u32();
                let elem = T::as_elem(scope);
                scope.create_local_array(Item::new(elem), size).into()
            })
        }
    }

    impl<T: CubePrimitive + Clone> Array<T> {
        /// Create an array from data.
        #[allow(unused_variables)]
        pub fn from_data<C: CubePrimitive>(data: impl IntoIterator<Item = C>) -> Self {
            Array { _val: PhantomData }
        }

        /// Expand function of [from_data](Array::from_data).
        pub fn __expand_from_data<C: CubePrimitive>(
            scope: &mut Scope,
            data: ArrayData<C>,
        ) -> <Self as CubeType>::ExpandType {
            let var = scope.create_const_array(Item::new(T::as_elem(scope)), data.values);
            ExpandElementTyped::new(var)
        }
    }

    /// Type useful for the expand function of [from_data](Array::from_data).
    pub struct ArrayData<C> {
        values: Vec<Variable>,
        _ty: PhantomData<C>,
    }

    impl<C: CubePrimitive + Into<ExpandElementTyped<C>>, T: IntoIterator<Item = C>> From<T>
        for ArrayData<C>
    {
        fn from(value: T) -> Self {
            let values: Vec<Variable> = value
                .into_iter()
                .map(|value| {
                    let value: ExpandElementTyped<C> = value.into();
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

/// Module that contains the implementation details of the line_size function.
mod line {
    use crate::prelude::Line;

    use super::*;

    impl<P: CubePrimitive> Array<Line<P>> {
        /// Get the size of each line contained in the tensor.
        ///
        /// Same as the following:
        ///
        /// ```rust, ignore
        /// let size = tensor[0].size();
        /// ```
        pub fn line_size(&self) -> u32 {
            unexpanded!()
        }

        // Expand function of [size](Tensor::line_size).
        pub fn __expand_line_size(
            expand: <Self as CubeType>::ExpandType,
            scope: &mut Scope,
        ) -> u32 {
            expand.__expand_line_size_method(scope)
        }
    }

    impl<P: CubePrimitive> ExpandElementTyped<Array<Line<P>>> {
        /// Comptime version of [size](Array::line_size).
        pub fn line_size(&self) -> u32 {
            self.expand
                .item
                .vectorization
                .unwrap_or(NonZero::new(1).unwrap())
                .get() as u32
        }

        // Expand method of [size](Array::line_size).
        pub fn __expand_line_size_method(&self, _content: &mut Scope) -> u32 {
            self.line_size()
        }
    }
}

/// Module that contains the implementation details of vectorization functions.
///
/// TODO: Remove vectorization in favor of the line API.
mod vectorization {

    use super::*;

    #[cube]
    impl<T: CubePrimitive + Clone> Array<T> {
        #[allow(unused_variables)]
        pub fn vectorized(#[comptime] length: u32, #[comptime] vectorization_factor: u32) -> Self {
            intrinsic!(|scope| {
                scope
                    .create_local_array(
                        Item::vectorized(
                            T::as_elem(scope),
                            NonZero::new(vectorization_factor as u8),
                        ),
                        length,
                    )
                    .into()
            })
        }

        #[allow(unused_variables)]
        pub fn to_vectorized(self, #[comptime] vectorization_factor: u32) -> T {
            intrinsic!(|scope| {
                let factor = vectorization_factor;
                let var = self.expand.clone();
                let item = Item::vectorized(var.item.elem(), NonZero::new(factor as u8));

                let new_var = if factor == 1 {
                    let new_var = scope.create_local(item);
                    let element = index::expand(
                        scope,
                        self.clone(),
                        ExpandElementTyped::from_lit(scope, 0u32),
                    );
                    assign::expand::<T>(scope, element, new_var.clone().into());
                    new_var
                } else {
                    let new_var = scope.create_local_mut(item);
                    for i in 0..factor {
                        let expand: Self = self.expand.clone().into();
                        let element =
                            index::expand(scope, expand, ExpandElementTyped::from_lit(scope, i));
                        index_assign::expand::<ExpandElementTyped<Array<T>>, T>(
                            scope,
                            new_var.clone().into(),
                            ExpandElementTyped::from_lit(scope, i),
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
        pub fn len(&self) -> u32 {
            intrinsic!(|scope| {
                ExpandElement::Plain(expand_length_native(scope, *self.expand)).into()
            })
        }

        /// Obtain the array buffer length
        pub fn buffer_len(&self) -> u32 {
            intrinsic!(|scope| {
                let out = scope.create_local(Item::new(u32::as_elem(scope)));
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

    use crate::{
        ir::Instruction,
        prelude::{CubeIndex, CubeIndexMut},
    };

    use super::*;

    #[cube]
    impl<E: CubePrimitive> Array<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&self, i: u32) -> &E
        where
            Self: CubeIndex,
        {
            intrinsic!(|scope| {
                let out = scope.create_local(self.expand.item);
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: *self.expand,
                        index: i.expand.consume(),
                        line_size: 0,
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
        pub unsafe fn index_assign_unchecked(&mut self, i: u32, value: E)
        where
            Self: CubeIndexMut,
        {
            intrinsic!(|scope| {
                scope.register(Instruction::new(
                    Operator::UncheckedIndexAssign(IndexAssignOperator {
                        index: i.expand.consume(),
                        value: value.expand.consume(),
                        line_size: 0,
                    }),
                    *self.expand,
                ));
            })
        }
    }
}

impl<C: CubeType> CubeType for Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> CubeType for &Array<C> {
    type ExpandType = ExpandElementTyped<Array<C>>;
}

impl<C: CubeType> ExpandElementBaseInit for Array<C> {
    fn init_elem(_scope: &mut crate::ir::Scope, elem: ExpandElement) -> ExpandElement {
        // The type can't be deeply cloned/copied.
        elem
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
        this: ExpandElementTyped<Array<T>>,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, this, idx)
    }
}

impl<T: CubePrimitive> ListExpand<T> for ExpandElementTyped<Array<T>> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index_unchecked::expand(scope, self.clone(), idx)
    }
}

impl<T: CubePrimitive> ListMut<T> for Array<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: ExpandElementTyped<Array<T>>,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for ExpandElementTyped<Array<T>> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
