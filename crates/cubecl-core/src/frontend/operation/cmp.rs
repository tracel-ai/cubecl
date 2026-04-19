use core::cmp::Ordering;

use cubecl_ir::{Arithmetic, ClampOperator};

use crate as cubecl;
use crate::frontend::NativeExpand;
use crate::ir::{Comparison, Scope};
use crate::prelude::*;

// NOTE: Unary comparison tests are in the unary module

pub trait CubeEq: Eq + CubePrimitive + CubeType<ExpandType: EqExpand> + Sized {
    fn __expand_eq(
        scope: &mut Scope,
        lhs: &NativeExpand<Self>,
        rhs: &NativeExpand<Self>,
    ) -> NativeExpand<bool> {
        lhs.__expand_eq_method(scope, rhs)
    }
    fn __expand_ne(
        scope: &mut Scope,
        lhs: &NativeExpand<Self>,
        rhs: &NativeExpand<Self>,
    ) -> NativeExpand<bool> {
        lhs.__expand_ne_method(scope, rhs)
    }
}
pub trait EqExpand {
    fn __expand_eq_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_ne_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool>;
}
impl<T: Eq + CubePrimitive> CubeEq for T {}

impl<T: Eq + CubePrimitive> EqExpand for NativeExpand<T> {
    fn __expand_eq_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        cmp_expand(scope, this.into(), rhs.into(), Comparison::Equal).into()
    }
    fn __expand_ne_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        cmp_expand(scope, this.into(), rhs.into(), Comparison::NotEqual).into()
    }
}

#[derive_expand(CubeType, CubeTypeMut, IntoRuntime)]
#[cube(runtime_variants, no_constructors)]
pub enum Ordering {
    Less = -1,
    Equal = 0,
    Greater = 1,
}

fn ordering_disc(name: &'static str) -> NativeExpand<i32> {
    OrderingExpand::discriminant_of(name).into()
}

#[allow(non_snake_case)]
pub trait CubeOrdering {
    fn Less() -> Ordering {
        Ordering::Less
    }
    fn Equal() -> Ordering {
        Ordering::Equal
    }
    fn Greater() -> Ordering {
        Ordering::Greater
    }
    fn __expand_Less(_scope: &mut Scope) -> OrderingExpand {
        OrderingExpand {
            discriminant: ordering_disc("Less"),
            value: (),
        }
    }
    fn __expand_Equal(_scope: &mut Scope) -> OrderingExpand {
        OrderingExpand {
            discriminant: ordering_disc("Equal"),
            value: (),
        }
    }
    fn __expand_Greater(_scope: &mut Scope) -> OrderingExpand {
        OrderingExpand {
            discriminant: ordering_disc("Greater"),
            value: (),
        }
    }
}

impl CubeOrdering for Ordering {}

pub trait CubeOrd: Ord + CubeType<ExpandType: OrdExpand> + Sized {
    fn __expand_cmp(
        scope: &mut Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> OrderingExpand {
        lhs.__expand_cmp_method(scope, rhs)
    }

    fn __expand_min(
        scope: &mut Scope,
        lhs: Self::ExpandType,
        rhs: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_min_method(scope, rhs)
    }

    fn __expand_max(
        scope: &mut Scope,
        lhs: Self::ExpandType,
        rhs: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_max_method(scope, rhs)
    }

    fn __expand_clamp(
        scope: &mut Scope,
        lhs: Self::ExpandType,
        min: Self::ExpandType,
        max: Self::ExpandType,
    ) -> Self::ExpandType {
        lhs.__expand_clamp_method(scope, min, max)
    }
}
pub trait OrdExpand {
    fn __expand_cmp_method(&self, scope: &mut Scope, rhs: &Self) -> OrderingExpand;
    fn __expand_min_method(self, scope: &mut Scope, rhs: Self) -> Self;
    fn __expand_max_method(self, scope: &mut Scope, rhs: Self) -> Self;
    fn __expand_clamp_method(self, scope: &mut Scope, min: Self, max: Self) -> Self;
}

impl<T: Ord + CubePrimitive> CubeOrd for T {}
impl<T: Ord + CubePrimitive> OrdExpand for NativeExpand<T> {
    fn __expand_cmp_method(&self, scope: &mut Scope, rhs: &Self) -> OrderingExpand {
        let lhs_lt_rhs = self.__expand_lt_method(scope, rhs);
        let lhs_gt_rhs = self.__expand_gt_method(scope, rhs);
        let less = ordering_disc("Less");
        let equal = ordering_disc("Equal");
        let greater = ordering_disc("Greater");
        let eq_or_gt = select::expand(scope, lhs_gt_rhs, greater, equal);
        let discriminant = select::expand(scope, lhs_lt_rhs, less, eq_or_gt);
        OrderingExpand {
            discriminant,
            value: (),
        }
    }
    fn __expand_min_method(self, scope: &mut Scope, rhs: Self) -> Self {
        binary_expand(scope, self.into(), rhs.into(), Arithmetic::Min).into()
    }
    fn __expand_max_method(self, scope: &mut Scope, rhs: Self) -> Self {
        binary_expand(scope, self.into(), rhs.into(), Arithmetic::Max).into()
    }
    fn __expand_clamp_method(self, scope: &mut Scope, min: Self, max: Self) -> Self {
        unary_expand(scope, self.into(), |op| {
            Arithmetic::Clamp(ClampOperator {
                input: op.input,
                min_value: min.expand,
                max_value: max.expand,
            })
        })
        .into()
    }
}

pub trait CubePartialOrd: PartialOrd + CubeType<ExpandType: PartialOrdExpand> + Sized {
    fn __expand_partial_cmp(
        scope: &mut Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> OptionExpand<Ordering> {
        lhs.__expand_partial_cmp_method(scope, rhs)
    }

    fn __expand_lt(
        scope: &mut Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_lt_method(scope, rhs)
    }

    fn __expand_le(
        scope: &mut Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_le_method(scope, rhs)
    }

    fn __expand_gt(
        scope: &mut Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_gt_method(scope, rhs)
    }

    fn __expand_ge(
        scope: &mut Scope,
        lhs: &Self::ExpandType,
        rhs: &Self::ExpandType,
    ) -> NativeExpand<bool> {
        lhs.__expand_ge_method(scope, rhs)
    }
}
pub trait PartialOrdExpand {
    fn __expand_partial_cmp_method(&self, scope: &mut Scope, rhs: &Self) -> OptionExpand<Ordering>;
    fn __expand_lt_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_le_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_gt_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool>;
    fn __expand_ge_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool>;
}

impl<T: PartialOrd + CubePrimitive> CubePartialOrd for T {}
impl<T: PartialOrd + CubePrimitive> PartialOrdExpand for NativeExpand<T> {
    fn __expand_partial_cmp_method(&self, scope: &mut Scope, rhs: &Self) -> OptionExpand<Ordering> {
        let lhs_lt_rhs = self.__expand_lt_method(scope, rhs);
        let lhs_gt_rhs = self.__expand_gt_method(scope, rhs);
        let less = ordering_disc("Less");
        let equal = ordering_disc("Equal");
        let greater = ordering_disc("Greater");
        let eq_or_gt = select::expand(scope, lhs_gt_rhs, greater, equal);
        let discriminant = select::expand(scope, lhs_lt_rhs, less, eq_or_gt);
        Option::__expand_new_Some(
            scope,
            OrderingExpand {
                discriminant,
                value: (),
            },
        )
    }
    fn __expand_lt_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        cmp_expand(scope, this.into(), rhs.into(), Comparison::Lower).into()
    }
    fn __expand_le_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        cmp_expand(scope, this.into(), rhs.into(), Comparison::LowerEqual).into()
    }
    fn __expand_gt_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        cmp_expand(scope, this.into(), rhs.into(), Comparison::Greater).into()
    }
    fn __expand_ge_method(&self, scope: &mut Scope, rhs: &Self) -> NativeExpand<bool> {
        let this = self.__expand_deref_method(scope);
        let rhs = rhs.__expand_deref_method(scope);
        cmp_expand(scope, this.into(), rhs.into(), Comparison::GreaterEqual).into()
    }
}
