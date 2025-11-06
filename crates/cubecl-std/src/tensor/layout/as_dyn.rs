use cubecl::prelude::*;
use cubecl_core::{self as cubecl, unexpanded};
use variadics_please::all_tuples;

use crate::tensor::layout::*;

/// Coordinates that can be converted to a dynamic sequence of signed coordinates.
/// Can be used to convert any set of coordinates to a comptime-sized sequence for use with TMA.
#[cube]
pub trait IntoDyn: Coordinates + LaunchArg {
    fn into_dyn(self) -> Sequence<i32> {
        unexpanded!()
    }
}

macro_rules! as_ty {
    ($T: ident, $dummy: ident) => {
        $T
    };
}

macro_rules! impl_tuple {
    ($ty: ident, $($t: ident),*) => {
        impl IntoDyn for ($(as_ty!($ty, $t)),*) {}

        impl IntoDynExpand for ($(ExpandElementTyped<as_ty!($ty, $t)>),*) {
            fn __expand_into_dyn_method(self, scope: &mut Scope) -> SequenceExpand<i32> {
                let mut seq = Sequence::__expand_new(scope);
                let ($($t),*) = self;
                let ($($t),*) = ($(i32::__expand_cast_from(scope, $t)),*);
                $(seq.__expand_push_method(scope, $t);)*
                seq
            }
        }
    };
}

macro_rules! impl_tuples {
    ($($t: ident),*) => {
        impl_tuple!(u32, $($t),*);
        impl_tuple!(i32, $($t),*);
    };
}

all_tuples!(impl_tuples, 2, 12, t);

#[cube]
impl IntoDyn for Sequence<i32> {
    fn into_dyn(self) -> Sequence<i32> {
        self
    }
}

#[cube]
impl IntoDyn for Sequence<u32> {
    fn into_dyn(self) -> Sequence<i32> {
        let mut seq = Sequence::new();
        for x in self {
            seq.push(i32::cast_from(x));
        }
        seq
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct IntoDynLayout<L: Layout<SourceCoordinates: IntoDyn> + LaunchArg> {
    layout: L,
}

impl<L: Layout<SourceCoordinates: IntoDyn> + LaunchArg> IntoDynLayout<L> {
    pub fn new(layout: L) -> Self {
        IntoDynLayout { layout }
    }
}

#[cube]
impl<L: Layout<SourceCoordinates: IntoDyn> + LaunchArg> Layout for IntoDynLayout<L> {
    type Coordinates = L::Coordinates;
    type SourceCoordinates = Sequence<i32>;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let pos = self.layout.to_source_pos(pos);
        pos.into_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.layout.is_in_bounds(pos)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let (pos, in_bounds) = self.layout.to_source_pos_checked(pos);
        (pos.into_dyn(), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.layout.shape()
    }
}
