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

macro_rules! impl_tuple {
    ($(($T: ident, $t: ident)),*) => {
        impl<$($T: Coordinates + CubePrimitive + LaunchArg),*> IntoDyn for ($($T),*) {}

        impl<$($T: Coordinates + CubePrimitive + LaunchArg),*> IntoDynExpand for ($(ExpandElementTyped<$T>),*) {
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

all_tuples!(impl_tuple, 2, 12, T, t);

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

#[derive(CubeType, CubeLaunch)]
pub struct IntoDyn2Layout<L: Layout<SourceCoordinates = (P, O)> + LaunchArg, P: IntoDyn, O: IntoDyn>
{
    layout: L,
}

impl<L: Layout<SourceCoordinates: IntoDyn> + LaunchArg> IntoDynLayout<L> {
    pub fn new(layout: L) -> Self {
        IntoDynLayout { layout }
    }
}

impl<L: Layout<SourceCoordinates = (P, O)> + LaunchArg, P: IntoDyn, O: IntoDyn + LaunchArg>
    IntoDyn2Layout<L, P, O>
{
    pub fn new(layout: L) -> Self {
        IntoDyn2Layout { layout }
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

#[cube]
impl<L: Layout<SourceCoordinates = (P, O)> + LaunchArg, P: IntoDyn, O: IntoDyn + LaunchArg> Layout
    for IntoDyn2Layout<L, P, O>
{
    type Coordinates = L::Coordinates;
    type SourceCoordinates = (Sequence<i32>, Sequence<i32>);

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let pos = self.layout.to_source_pos(pos);
        (pos.0.into_dyn(), pos.1.into_dyn())
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.layout.is_in_bounds(pos)
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let (pos, in_bounds) = self.layout.to_source_pos_checked(pos);
        ((pos.0.into_dyn(), pos.1.into_dyn()), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.layout.shape()
    }
}
