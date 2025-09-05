use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, unexpanded};

use crate::tensor::layout::{Coordinates, Layout};

/// A virtual layout, to carry a layout without the need for generic parameters everywhere.
/// `C` represents the coordinate space of the underlying layout.
#[derive(Clone)]
pub struct VirtualLayout<C: Coordinates, S: Coordinates> {
    _coords: PhantomData<(C, S)>,
}

impl<C: Coordinates, S: Coordinates> Copy for VirtualLayout<C, S> {}

#[derive(Clone)]
pub struct VirtualLayoutExpand<C: Coordinates, S: Coordinates> {
    pub(crate) state: Arc<dyn VirtualLayoutOperationsExpand<C, S>>,
}

#[cube]
impl<C: Coordinates, S: Coordinates> VirtualLayout<C, S> {
    /// Virtual version of [Layout::to_source_pos]
    #[allow(unused)]
    pub fn to_source_pos(&self, pos: C) -> S {
        intrinsic!(|scope| { self.state.__expand_to_source_pos_method(scope, pos) })
    }

    /// Virtual version of [Layout::to_source_pos_checked]
    #[allow(unused)]
    pub fn to_source_pos_checked(&self, pos: C) -> (S, bool) {
        intrinsic!(|scope| { self.state.__expand_to_source_pos_checked_method(scope, pos) })
    }

    /// Virtual version of [Layout::shape]
    pub fn shape(&self) -> C {
        intrinsic!(|scope| { self.state.__expand_shape_method(scope) })
    }

    /// Virtual version of [Layout::is_in_bounds]
    #[allow(unused)]
    pub fn is_in_bounds(&self, pos: C) -> bool {
        intrinsic!(|scope| { self.state.__expand_is_in_bounds_method(scope, pos) })
    }
}

impl<C: Coordinates, S: Coordinates> VirtualLayout<C, S> {
    /// Create a new virtual layout from a concrete one
    pub fn new<L>(_layout: L) -> VirtualLayout<C, S>
    where
        L: VirtualLayoutOperations<C, S> + CubeType,
        L::ExpandType: VirtualLayoutOperationsExpand<C, S>,
    {
        unexpanded!()
    }

    /// Expand function of [VirtualLayout::__expand_new]
    pub fn __expand_new<L>(_scope: &mut Scope, layout: L::ExpandType) -> VirtualLayoutExpand<C, S>
    where
        L: VirtualLayoutOperations<C, S> + CubeType,
        L::ExpandType: VirtualLayoutOperationsExpand<C, S> + 'static,
    {
        VirtualLayoutExpand::new::<L>(layout)
    }
}

impl<C: Coordinates, S: Coordinates> VirtualLayoutExpand<C, S> {
    /// Create a new virtual layout from a concrete one
    pub fn new<L>(layout: L::ExpandType) -> VirtualLayoutExpand<C, S>
    where
        L: VirtualLayoutOperations<C, S> + CubeType,
        L::ExpandType: VirtualLayoutOperationsExpand<C, S> + 'static,
    {
        VirtualLayoutExpand::<C, S> {
            state: Arc::new(layout),
        }
    }
}

impl<C: Coordinates, S: Coordinates> CubeType for VirtualLayout<C, S> {
    type ExpandType = VirtualLayoutExpand<C, S>;
}

impl<C: Coordinates, S: Coordinates> IntoMut for VirtualLayoutExpand<C, S> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<C: Coordinates, S: Coordinates> CubeDebug for VirtualLayoutExpand<C, S> {}

/// Virtualized layout
pub trait VirtualLayoutOperations<C: Coordinates, S: Coordinates> {
    fn to_source_pos(&self, pos: C) -> S;
    fn to_source_pos_checked(&self, pos: C) -> (S, bool);
    fn shape(&self) -> C;
    fn is_in_bounds(&self, pos: C) -> bool;
}

/// Expand for virtualized layouts. Should be implemented for layouts to make them compatible with
/// [VirtualLayout] and [TensorView].
pub trait VirtualLayoutOperationsExpand<C: Coordinates, S: Coordinates> {
    fn __expand_to_source_pos_method(&self, scope: &mut Scope, pos: C::ExpandType)
    -> S::ExpandType;
    fn __expand_to_source_pos_checked_method(
        &self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> <(S, bool) as CubeType>::ExpandType;
    fn __expand_shape_method(&self, scope: &mut Scope) -> C::ExpandType;
    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<bool>;
}

impl<L: Layout> VirtualLayoutOperations<L::Coordinates, L::SourceCoordinates> for L {
    fn to_source_pos(&self, pos: L::Coordinates) -> L::SourceCoordinates {
        L::to_source_pos(self, pos)
    }

    fn to_source_pos_checked(&self, pos: L::Coordinates) -> (L::SourceCoordinates, bool) {
        L::to_source_pos_checked(self, pos)
    }

    fn shape(&self) -> L::Coordinates {
        L::shape(self)
    }

    fn is_in_bounds(&self, pos: L::Coordinates) -> bool {
        L::is_in_bounds(self, pos)
    }
}
