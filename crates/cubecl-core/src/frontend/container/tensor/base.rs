use crate::frontend::{ExpandElementBaseInit, ExpandElementTyped, SizedContainer};
use crate::{
    frontend::{indexation::Index, CubeContext, CubePrimitive, CubeType, ExpandElement},
    ir::{Elem, Item, Metadata, Variable},
    prelude::Line,
    unexpanded,
};
use std::{marker::PhantomData, num::NonZero};

/// The tensor type is similar to the [array type](crate::prelude::Array), however it comes with more
/// metadata such as [stride](Tensor::stride) and [shape](Tensor::shape).
#[derive(new)]
pub struct Tensor<T: CubeType> {
    _val: PhantomData<T>,
}

impl<T: CubeType> CubeType for Tensor<T> {
    type ExpandType = ExpandElementTyped<Tensor<T>>;
}

impl<C: CubeType> ExpandElementBaseInit for Tensor<C> {
    fn init_elem(_context: &mut crate::prelude::CubeContext, elem: ExpandElement) -> ExpandElement {
        // The type can't be deeply cloned/copied.
        elem
    }
}

impl<T: CubeType> Tensor<T> {
    /// Obtain the stride of input at dimension dim
    pub fn stride<C: Index>(&self, _dim: C) -> u32 {
        unexpanded!()
    }

    /// Obtain the shape of input at dimension dim
    pub fn shape<C: Index>(&self, _dim: C) -> u32 {
        unexpanded!()
    }

    /// The length of the buffer representing the tensor.
    ///
    /// # Warning
    ///
    /// The length will be affected by the vectorization factor. To obtain the number of elements,
    /// you should multiply the length by the vectorization factor.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unexpanded!()
    }

    /// Returns the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unexpanded!()
    }
}

impl<T: CubeType> ExpandElementTyped<T> {
    // Expanded version of stride
    pub fn __expand_stride_method(
        self,
        context: &mut CubeContext,
        dim: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        let dim: ExpandElement = dim.into();
        let out = context.create_local_binding(Item::new(Elem::UInt));
        context.register(Metadata::Stride {
            dim: *dim,
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out.into()
    }

    // Expanded version of shape
    pub fn __expand_shape_method(
        self,
        context: &mut CubeContext,
        dim: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        let dim: ExpandElement = dim.into();
        let out = context.create_local_binding(Item::new(Elem::UInt));
        context.register(Metadata::Shape {
            dim: *dim,
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out.into()
    }

    // Expanded version of len
    pub fn __expand_len_method(self, context: &mut CubeContext) -> ExpandElementTyped<u32> {
        let var: Variable = *self.expand;

        // In some case the len expand should return the vectorization factor.
        let item = match var {
            Variable::Local { item, .. } => Some(item),
            Variable::LocalBinding { item, .. } => Some(item),
            _ => None,
        };

        if let Some(val) = item {
            let var = Variable::ConstantScalar(crate::ir::ConstantScalarValue::UInt(
                val.vectorization.map(|val| val.get() as u64).unwrap_or(1),
            ));

            return ExpandElement::Plain(var).into();
        };

        let out = context.create_local_binding(Item::new(Elem::UInt));
        context.register(Metadata::Length {
            var: self.expand.into(),
            out: out.clone().into(),
        });
        out.into()
    }

    // Expanded version of rank.
    pub fn __expand_rank_method(self, _context: &mut CubeContext) -> ExpandElementTyped<u32> {
        ExpandElement::Plain(Variable::Rank).into()
    }
}

impl<T: CubeType<ExpandType = ExpandElementTyped<T>>> SizedContainer for Tensor<T> {
    type Item = T;
}

impl<T: CubeType> Iterator for &Tensor<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

impl<P: CubePrimitive> Tensor<Line<P>> {
    pub fn line_size(&self) -> u32 {
        unexpanded!()
    }

    pub fn __expand_line_size(
        expand: <Self as CubeType>::ExpandType,
        context: &mut CubeContext,
    ) -> u32 {
        expand.__expand_line_size_method(context)
    }
}

impl<P: CubePrimitive> ExpandElementTyped<Tensor<Line<P>>> {
    // So that it can be used in comptime.
    pub fn line_size(&self) -> u32 {
        self.expand
            .item()
            .vectorization
            .clone()
            .unwrap_or(NonZero::new(1).unwrap())
            .get() as u32
    }
    pub fn __expand_line_size_method(&self, _content: &mut CubeContext) -> u32 {
        self.line_size()
    }
}
