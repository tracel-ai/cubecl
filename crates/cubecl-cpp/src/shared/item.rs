use std::fmt::Display;

use super::{Dialect, Elem};

#[derive(Debug, Clone, PartialEq, Eq, Copy, Hash)]
pub struct Item<D: Dialect> {
    pub(crate) elem: Elem<D>,
    pub(crate) vectorization: usize,
    pub(crate) native: bool,
}

impl<D: Dialect> Display for Item<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_item(f, &self)
    }
}

impl<D: Dialect> Item<D> {
    pub fn elem(&self) -> &Elem<D> {
        &self.elem
    }

    pub fn de_optimized(&self) -> Self {
        match self.elem {
            Elem::F162 => Item::new(Elem::F16, self.vectorization * 2, self.native),
            Elem::BF162 => Item::new(Elem::BF16, self.vectorization * 2, self.native),
            _ => *self,
        }
    }

    pub fn new(elem: Elem<D>, vectorization: usize, native: bool) -> Self {
        Self {
            elem,
            vectorization,
            native,
        }
    }
    pub fn scalar(elem: Elem<D>, native: bool) -> Self {
        Self {
            elem,
            vectorization: 1,
            native,
        }
    }

    pub fn is_optimized(&self) -> bool {
        matches!(self.elem, Elem::F162 | Elem::BF162)
    }

    pub fn optimized(&self) -> Item<D> {
        if self.vectorization == 1 {
            return *self;
        }

        if self.vectorization % 2 != 0 {
            return *self;
        }

        match self.elem {
            Elem::F16 => Item {
                elem: Elem::F162,
                vectorization: self.vectorization / 2,
                native: self.native,
            },
            Elem::BF16 => Item {
                elem: Elem::BF162,
                vectorization: self.vectorization / 2,
                native: self.native,
            },
            _ => *self,
        }
    }
}
