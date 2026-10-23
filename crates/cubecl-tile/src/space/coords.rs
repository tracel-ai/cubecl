//! A coordinate or extent list whose entries keep their constness.

use cubecl::ir::{ExpandValue, Scope};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::layout::CoordsDyn;

use crate::algebra::{Integer, IntegerExpand, simplified_add, simplified_mul};

/// An immutable coordinate/extent list whose expand's `IntoMut` is the identity.
/// Copying into mutable slots (as `Sequence` does) would erase constness.
pub struct Coords<C: Int> {
    _c: core::marker::PhantomData<C>,
}

impl<C: Int> Clone for Coords<C> {
    fn clone(&self) -> Self {
        Coords {
            _c: core::marker::PhantomData,
        }
    }
}

#[allow(clippy::new_without_default, clippy::len_without_is_empty)]
impl<C: Int> Coords<C> {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> Self {
        unexpanded!()
    }
    pub fn push(&mut self, _v: C) {
        unexpanded!()
    }
    /// The element at comptime `i`.
    pub fn at(&self, _i: usize) -> C {
        unexpanded!()
    }
    /// The comptime length.
    pub fn len(&self) -> usize {
        unexpanded!()
    }
    /// Re-view as boundary [`CoordsDyn`].
    pub(crate) fn to_dyn(&self) -> CoordsDyn {
        unexpanded!()
    }
    /// Product of the elements at comptime `picks` (empty picks fold to `1`).
    pub(crate) fn product(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }
    /// Sum of the elements at comptime `picks` (empty picks fold to `0`).
    pub(crate) fn sum(&self, _picks: Vec<usize>) -> C {
        unexpanded!()
    }

    /// Copy these coordinates into mutable kernel registers.
    pub(crate) fn stored(&self) -> Coords<C> {
        unexpanded!()
    }

    /// Assign every coordinate into this carrier, produced by [`stored`](Coords::stored).
    pub(crate) fn store_from(&mut self, _src: &Coords<C>) {
        unexpanded!()
    }

    pub fn __expand_new(_scope: &Scope) -> CoordsExpand<C> {
        CoordsExpand { values: Vec::new() }
    }
}

#[cube]
impl Coords<u32> {
    /// Constant coordinates holding comptime `values`.
    // `#[unroll]` needs a range loop; an iterator has no expansion.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn constant(#[comptime] values: Vec<usize>) -> Coords<u32> {
        let mut out = Coords::<u32>::new();

        #[unroll]
        for p in 0..comptime!(values.len()) {
            out.push(comptime!(values[p] as u32).runtime());
        }

        out
    }

    /// The digits of a flat row-major index `i` over these extents.
    /// The leading entry skips the modulo so the divide folds for constant extents.
    pub(crate) fn unravel(&self, i: u32) -> Coords<u32> {
        let n = self.len();
        let mut out = Coords::<u32>::new();

        #[unroll]
        for p in 0..n {
            let digit = i.divided_by(self.product(comptime!(((p + 1)..n).collect::<Vec<_>>())));
            if comptime!(p == 0) {
                out.push(digit);
            } else {
                out.push(digit.remainder(self.at(p)));
            }
        }

        out
    }

    /// Append `other`'s coordinates after these.
    pub(crate) fn extend(&mut self, other: &Coords<u32>) {
        #[unroll]
        for p in 0..other.len() {
            self.push(other.at(p));
        }
    }

    /// Whether every coordinate of `pos` falls inside these extents.
    pub(crate) fn within(&self, pos: CoordsDyn) -> bool {
        let mut valid = true;

        #[unroll]
        for p in 0..self.len() {
            valid = valid && pos[p] < self.at(p);
        }

        valid
    }
}

pub struct CoordsExpand<C: Int> {
    values: Vec<NativeExpand<C>>,
}

impl<C: Int> CubeType for Coords<C> {
    type ExpandType = CoordsExpand<C>;
}

impl<C: Int> IntoExpand for CoordsExpand<C> {
    type Expand = Self;
    fn into_expand(self, _scope: &Scope) -> Self {
        self
    }
}

/// Identity, to keep constness (see [`Coords`]).
impl<C: Int> IntoMut for CoordsExpand<C> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<C: Int> CubeDebug for CoordsExpand<C> {}

impl<C: Int> Clone for CoordsExpand<C> {
    fn clone(&self) -> Self {
        CoordsExpand {
            values: self.values.clone(),
        }
    }
}

impl<C: Int> ExpandTypeClone for CoordsExpand<C> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl<C: Int> AsRefExpand for CoordsExpand<C> {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}

impl<C: Int> AsMutExpand for CoordsExpand<C> {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}

impl<C: Int> CoordsExpand<C> {
    /// The entries known only at runtime, in order.
    pub(crate) fn runtime_mut(&mut self) -> impl Iterator<Item = &mut NativeExpand<C>> {
        self.values
            .iter_mut()
            .filter(|v| matches!(v.expand, ExpandValue::Value(_)))
    }

    pub fn __expand_assign_method(&mut self, _scope: &Scope, other: Self) {
        self.values = other.values;
    }
    pub fn __expand_push_method(&mut self, _scope: &Scope, v: NativeExpand<C>) {
        self.values.push(v);
    }
    pub fn __expand_at_method(&self, _scope: &Scope, i: NativeExpand<usize>) -> NativeExpand<C> {
        let i = i
            .expand
            .as_const()
            .expect("Coords::at: comptime index only")
            .as_i64() as usize;
        self.values[i]
    }
    pub fn __expand_len_method(&self, _scope: &Scope) -> usize {
        self.values.len()
    }
    pub fn __expand_to_dyn_method(&self, scope: &Scope) -> SequenceExpand<u32> {
        let mut out = Sequence::<u32>::__expand_new(scope);
        for v in &self.values {
            out.__expand_push_method(scope, unsafe { *v.as_type_ref_unchecked::<u32>() });
        }
        out
    }
    pub fn __expand_product_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            ExpandValue::constant(1u64.into(), C::elem_type(scope)).into();
        for i in picks {
            acc = simplified_mul(scope, acc, self.values[i]);
        }
        acc
    }
    pub fn __expand_sum_method(&self, scope: &Scope, picks: Vec<usize>) -> NativeExpand<C> {
        let mut acc: NativeExpand<C> =
            ExpandValue::constant(0u64.into(), C::elem_type(scope)).into();
        for i in picks {
            acc = simplified_add(scope, acc, self.values[i]);
        }
        acc
    }

    pub fn __expand_stored_method(&self, scope: &Scope) -> CoordsExpand<C> {
        CoordsExpand {
            values: self.values.iter().map(|v| (*v).into_mut(scope)).collect(),
        }
    }

    pub fn __expand_store_from_method(&mut self, scope: &Scope, src: &CoordsExpand<C>) {
        assert_eq!(
            self.values.len(),
            src.values.len(),
            "Coords::store_from: source and destination lengths differ"
        );
        for (dst, src) in self.values.iter_mut().zip(&src.values) {
            dst.__expand_assign_method(scope, *src);
        }
    }
}
