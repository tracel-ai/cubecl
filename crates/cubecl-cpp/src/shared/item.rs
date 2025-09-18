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
        D::compile_item(f, self)
    }
}

impl<D: Dialect> Item<D> {
    pub fn elem(&self) -> &Elem<D> {
        &self.elem
    }

    pub const fn size(&self) -> usize {
        self.elem.size() * self.vectorization
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

    pub fn can_be_optimized(&self) -> bool {
        D::item_can_be_optimized()
    }

    pub fn is_optimized(&self) -> bool {
        matches!(
            self.elem,
            Elem::F16x2 | Elem::BF16x2 | Elem::FP4x2(_) | Elem::FP6x2(_) | Elem::FP8x2(_)
        )
    }

    pub fn optimized(&self) -> Item<D> {
        if !self.can_be_optimized() || !self.vectorization.is_multiple_of(2) {
            return *self;
        }

        match self.elem {
            Elem::F16 => Item {
                elem: Elem::F16x2,
                vectorization: self.vectorization / 2,
                native: self.native,
            },
            Elem::BF16 => Item {
                elem: Elem::BF16x2,
                vectorization: self.vectorization / 2,
                native: self.native,
            },
            Elem::FP4(kind) => Item {
                elem: Elem::FP4x2(kind),
                vectorization: self.vectorization / 2,
                native: self.native,
            },
            Elem::FP6(kind) => Item {
                elem: Elem::FP6x2(kind),
                vectorization: self.vectorization / 2,
                native: self.native,
            },
            Elem::FP8(kind) => Item {
                elem: Elem::FP8x2(kind),
                vectorization: self.vectorization / 2,
                native: self.native,
            },
            _ => *self,
        }
    }

    /// Get the number of values packed into a single storage element. (i.e. `f16x2 -> 2`)
    pub fn packing_factor(&self) -> usize {
        self.elem.packing_factor()
    }

    pub fn de_optimized(&self) -> Self {
        match self.elem {
            Elem::FP4x2(kind) => Item::new(Elem::FP4(kind), self.vectorization * 2, self.native),
            Elem::FP6x2(kind) => Item::new(Elem::FP6(kind), self.vectorization * 2, self.native),
            Elem::FP8x2(kind) => Item::new(Elem::FP8(kind), self.vectorization * 2, self.native),
            Elem::F16x2 => Item::new(Elem::F16, self.vectorization * 2, self.native),
            Elem::BF16x2 => Item::new(Elem::BF16, self.vectorization * 2, self.native),
            _ => *self,
        }
    }
}
