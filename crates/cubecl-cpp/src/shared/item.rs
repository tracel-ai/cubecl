use std::fmt::Display;

use cubecl_core::{ir::Intern, prelude::Visibility};

use super::{Dialect, Elem};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Item<D: Dialect> {
    Scalar(Elem<D>),
    Vector(Intern<Item<D>>, usize),
    NativeVector(Elem<D>, usize),
    Atomic(Intern<Item<D>>),
    Pointer(Intern<Item<D>>, PointerClass),
    Array(Intern<Item<D>>, usize),
    DynamicArray(Intern<Item<D>>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PointerClass {
    Global(Visibility),
    Shared,
    Local,
}

impl<D: Dialect> Display for Item<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        D::compile_item(f, self)
    }
}

impl<D: Dialect> Item<D> {
    pub fn new(inner: Elem<D>, vectorization: usize) -> Self {
        let scalar = Self::Scalar(inner);
        if vectorization > 1 {
            Self::Vector(scalar.intern(), vectorization)
        } else {
            scalar
        }
    }

    pub fn intern(self) -> Intern<Self> {
        Intern::new(self)
    }

    /// Type of the value, unwrapping pointers
    pub fn value_ty(&self) -> &Item<D> {
        match self {
            Item::Pointer(inner, _) => inner.value_ty(),
            Item::Array(inner, _) => inner.value_ty(),
            Item::DynamicArray(inner) => inner.value_ty(),
            other => other,
        }
    }

    pub fn elem(&self) -> &Elem<D> {
        match self {
            Item::Scalar(elem) | Item::NativeVector(elem, _) => elem,
            Item::Vector(item, _)
            | Item::Atomic(item)
            | Item::Pointer(item, _)
            | Item::Array(item, _)
            | Item::DynamicArray(item) => item.elem(),
        }
    }

    pub fn with_elem(&self, elem: Elem<D>) -> Self {
        match self {
            Item::Scalar(_) => Item::Scalar(elem),
            Item::NativeVector(_, vectorization) => Item::NativeVector(elem, *vectorization),
            Item::Vector(inner, vectorization) => {
                Item::Vector(inner.with_elem(elem).intern(), *vectorization)
            }
            Item::Atomic(inner) => Item::Atomic(inner.with_elem(elem).intern()),
            Item::Pointer(inner, class) => Item::Pointer(inner.with_elem(elem).intern(), *class),
            Item::Array(inner, size) => Item::Array(inner.with_elem(elem).intern(), *size),
            Item::DynamicArray(inner) => Item::DynamicArray(inner.with_elem(elem).intern()),
        }
    }

    pub fn as_scalar(&self) -> Self {
        match self {
            Item::Scalar(_) => *self,
            Item::NativeVector(elem, _) => Item::Scalar(*elem),
            Item::Vector(inner, _) => inner.as_scalar(),
            Item::Atomic(inner) => Item::Atomic(inner.as_scalar().intern()),
            Item::Pointer(inner, class) => Item::Pointer(inner.as_scalar().intern(), *class),
            Item::Array(inner, size) => Item::Array(inner.as_scalar().intern(), *size),
            Item::DynamicArray(inner) => Item::DynamicArray(inner.as_scalar().intern()),
        }
    }

    pub fn vectorization(&self) -> usize {
        match self {
            Item::Vector(_, vectorization) | Item::NativeVector(_, vectorization) => *vectorization,
            Item::Scalar(_) => 1,
            Item::Atomic(inner)
            | Item::Pointer(inner, _)
            | Item::Array(inner, _)
            | Item::DynamicArray(inner) => inner.vectorization(),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Item::Scalar(elem) => elem.size(),
            Item::Vector(inner, vectorization) => inner.size() * vectorization,
            Item::NativeVector(elem, vectorization) => elem.size() * vectorization,
            Item::Atomic(inner) => inner.size(),
            Item::Array(inner, size) => inner.size() * *size,
            Item::DynamicArray(inner) => inner.size(),
            Item::Pointer(..) => size_of::<u64>(),
        }
    }

    pub fn is_native(&self) -> bool {
        match self {
            Item::Scalar(..) | Item::NativeVector(..) => true,
            Item::Vector(..) => false,
            Item::Atomic(item)
            | Item::Pointer(item, ..)
            | Item::Array(item, _)
            | Item::DynamicArray(item) => item.is_native(),
        }
    }

    pub fn can_be_optimized(&self) -> bool {
        D::item_can_be_optimized()
    }

    pub fn is_optimized(&self) -> bool {
        matches!(
            self.elem(),
            Elem::F16x2 | Elem::BF16x2 | Elem::FP4x2(_) | Elem::FP6x2(_) | Elem::FP8x2(_)
        )
    }

    pub fn optimized(&self) -> Item<D> {
        if !self.can_be_optimized() {
            return *self;
        }

        match self {
            Item::Scalar(elem) => Item::Scalar(*elem),
            Item::Vector(inner, _) if !matches!(**inner, Item::Scalar(_)) => inner.optimized(),
            Item::Vector(inner, vectorization) => match Self::optimized_elem(*inner.elem()) {
                Some(elem) => Item::new(elem, *vectorization / elem.packing_factor()),
                None => Item::Vector(*inner, *vectorization),
            },
            Item::NativeVector(elem, vectorization) => match Self::optimized_elem(*elem) {
                Some(elem) if *vectorization > elem.packing_factor() => {
                    Item::NativeVector(elem, *vectorization / elem.packing_factor())
                }
                Some(elem) => Item::Scalar(elem),
                None => Item::NativeVector(*elem, *vectorization),
            },
            Item::Atomic(inner) => Item::Atomic(inner.optimized().intern()),
            Item::Pointer(inner, pointer_class) => {
                Item::Pointer(inner.optimized().intern(), *pointer_class)
            }
            Item::Array(inner, size) => Item::Array(inner.optimized().intern(), *size),
            Item::DynamicArray(inner) => Item::DynamicArray(inner.optimized().intern()),
        }
    }

    fn optimized_elem(elem: Elem<D>) -> Option<Elem<D>> {
        match elem {
            Elem::F16 => Some(Elem::F16x2),
            Elem::BF16 => Some(Elem::BF16x2),
            Elem::FP4(kind) => Some(Elem::FP4x2(kind)),
            Elem::FP6(kind) => Some(Elem::FP6x2(kind)),
            Elem::FP8(kind) => Some(Elem::FP8x2(kind)),
            _ => None,
        }
    }

    /// Get the number of values packed into a single storage element. (i.e. `f16x2 -> 2`)
    pub fn packing_factor(&self) -> usize {
        self.elem().packing_factor()
    }

    pub fn de_optimized(&self) -> Self {
        match self {
            Item::Scalar(elem) => Item::Scalar(*elem),
            Item::Vector(inner, _) if !matches!(**inner, Item::Scalar(_)) => inner.de_optimized(),
            Item::Vector(inner, vectorization) => match Self::deoptimized_elem(*inner.elem()) {
                Some(elem) => Item::Vector(
                    Item::Scalar(elem).intern(),
                    *vectorization * inner.elem().packing_factor(),
                ),
                None => Item::Vector(*inner, *vectorization),
            },
            Item::NativeVector(elem, vectorization) => match Self::deoptimized_elem(*elem) {
                Some(new_elem) => {
                    Item::NativeVector(new_elem, *vectorization * elem.packing_factor())
                }
                None => Item::NativeVector(*elem, *vectorization),
            },
            Item::Atomic(inner) => Item::Atomic(inner.de_optimized().intern()),
            Item::Pointer(inner, pointer_class) => {
                Item::Pointer(inner.de_optimized().intern(), *pointer_class)
            }
            Item::Array(inner, size) => Item::Array(inner.de_optimized().intern(), *size),
            Item::DynamicArray(inner) => Item::DynamicArray(inner.de_optimized().intern()),
        }
    }

    fn deoptimized_elem(elem: Elem<D>) -> Option<Elem<D>> {
        match elem {
            Elem::F16x2 => Some(Elem::F16),
            Elem::BF16x2 => Some(Elem::BF16),
            Elem::FP4x2(kind) => Some(Elem::FP4(kind)),
            Elem::FP6x2(kind) => Some(Elem::FP6(kind)),
            Elem::FP8x2(kind) => Some(Elem::FP8(kind)),
            _ => None,
        }
    }

    pub fn is_ptr(&self) -> bool {
        matches!(self, Item::Pointer(..))
    }

    pub fn is_const_ptr(&self) -> bool {
        matches!(
            self,
            Item::Pointer(
                _,
                PointerClass::Global(Visibility::Read | Visibility::ReadWrite)
            )
        )
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Item::Array(..) | Item::DynamicArray(..))
    }
}
