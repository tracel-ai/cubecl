use crate::frontend::{Array, CubeType, ExpandElement, Tensor};
use crate::unexpanded;

pub trait Vectorized {
    fn vectorization_factor(&self) -> u32;
    fn vectorize(self, factor: u32) -> Self;
}

impl<T: CubeType> Vectorized for Tensor<T> {
    fn vectorization_factor(&self) -> u32 {
        unexpanded!()
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Vectorized for &Tensor<T> {
    fn vectorization_factor(&self) -> u32 {
        unexpanded!()
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Vectorized for Array<T> {
    fn vectorization_factor(&self) -> u32 {
        unexpanded!()
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Vectorized for &Array<T> {
    fn vectorization_factor(&self) -> u32 {
        unexpanded!()
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl<T: CubeType> Vectorized for &mut Tensor<T> {
    fn vectorization_factor(&self) -> u32 {
        unexpanded!()
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl Vectorized for ExpandElement {
    fn vectorization_factor(&self) -> u32 {
        let var = match self {
            ExpandElement::Managed(var) => var,
            ExpandElement::Plain(var) => var,
        };

        var.item().vectorization.map(|it| it.get()).unwrap_or(1) as u32
    }

    fn vectorize(self, _factor: u32) -> Self {
        todo!()
    }
}

impl Vectorized for &ExpandElement {
    fn vectorization_factor(&self) -> u32 {
        let var = match self {
            ExpandElement::Managed(var) => var,
            ExpandElement::Plain(var) => var,
        };

        var.item().vectorization.map(|it| it.get()).unwrap_or(1) as u32
    }

    fn vectorize(self, _factor: u32) -> Self {
        todo!()
    }
}

/// Cubecl intrinsic. Gets the vectorization factor of an element at compile time.
pub fn vectorization_of<C: CubeType>(_element: &C) -> u32 {
    1
}

pub mod vectorization_of {
    use crate::prelude::*;

    pub fn expand<C: CubeType>(_context: &mut CubeContext, element: ExpandElementTyped<C>) -> u32 {
        let elem: ExpandElement = element.into();
        elem.item()
            .vectorization
            .map(|it| it.get() as u32)
            .unwrap_or(1)
    }
}
