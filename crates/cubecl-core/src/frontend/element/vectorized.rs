use crate::unexpanded;

use super::{CubeType, ExpandElement, Tensor};

pub trait IndexVec {
    fn idx(&self, idx: u32) -> &Self;
}

pub trait IndexVecMut: IndexVec {
    fn idx_mut(&mut self, _idx: u32) -> &mut Self;
}

pub trait Vectorized {
    fn vectorization_factor(&self) -> u32;
    fn vectorize(self, factor: u32) -> Self;
}

impl<T: Vectorized + CubeType> Vectorized for Tensor<T> {
    fn vectorization_factor(&self) -> u32 {
        unexpanded!()
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl<T: Vectorized + CubeType> Vectorized for &Tensor<T> {
    fn vectorization_factor(&self) -> u32 {
        unexpanded!()
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl<T: Vectorized + CubeType> Vectorized for &mut Tensor<T> {
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
