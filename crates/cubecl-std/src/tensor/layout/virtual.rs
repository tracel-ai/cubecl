use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, unexpanded};

use crate::tensor::layout::{Coordinates, Layout};

/// A virtual layout, to carry a layout without the need for generic parameters everywhere.
/// `C` represents the coordinate space of the underlying layout.
#[derive(Clone)]
pub struct VirtualLayout<C: Coordinates> {
    _coords: PhantomData<C>,
}

impl<C: Coordinates> Copy for VirtualLayout<C> {}

#[derive(Clone)]
pub struct VirtualLayoutExpand<C: Coordinates> {
    state: Arc<dyn VirtualLayoutOperationsExpand<C>>,
}

#[cube]
impl<C: Coordinates> VirtualLayout<C> {
    /// Virtual version of [Layout::to_linear_pos]
    #[allow(unused)]
    pub fn to_linear_pos(&self, pos: C) -> u32 {
        intrinsic!(|scope| { self.state.__expand_to_linear_pos_method(scope, pos) })
    }

    /// Virtual version of [Layout::to_linear_pos_checked]
    #[allow(unused)]
    pub fn to_linear_pos_checked(&self, pos: C) -> (u32, bool) {
        intrinsic!(|scope| { self.state.__expand_to_linear_pos_checked_method(scope, pos) })
    }

    /// Virtual version of [Layout::shape]
    pub fn shape(&self) -> C {
        intrinsic!(|scope| { self.state.__expand_shape_method(scope) })
    }
}

impl<C: Coordinates> VirtualLayout<C> {
    /// Create a new virtual layout from a concrete one
    pub fn new<L>(_layout: L) -> VirtualLayout<C>
    where
        L: VirtualLayoutOperations<C> + CubeType,
        L::ExpandType: VirtualLayoutOperationsExpand<C>,
    {
        unexpanded!()
    }

    /// Expand function of [VirtualLayout::__expand_new]
    pub fn __expand_new<L>(_scope: &mut Scope, layout: L::ExpandType) -> VirtualLayoutExpand<C>
    where
        L: VirtualLayoutOperations<C> + CubeType,
        L::ExpandType: VirtualLayoutOperationsExpand<C> + 'static,
    {
        VirtualLayoutExpand::<C> {
            state: Arc::new(layout),
        }
    }
}

impl<C: Coordinates> CubeType for VirtualLayout<C> {
    type ExpandType = VirtualLayoutExpand<C>;
}

impl<C: Coordinates> IntoMut for VirtualLayoutExpand<C> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<C: Coordinates> CubeDebug for VirtualLayoutExpand<C> {}

/// Virtualized layout
pub trait VirtualLayoutOperations<C: Coordinates> {
    fn to_linear_pos(&self, pos: C) -> u32;
    fn to_linear_pos_checked(&self, pos: C) -> (u32, bool);
    fn shape(&self) -> C;
}

/// Expand for virtualized layouts. Should be implemented for layouts to make them compatible with
/// [VirtualLayout] and [TensorView].
pub trait VirtualLayoutOperationsExpand<C: Coordinates> {
    fn __expand_to_linear_pos_method(
        &self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> <u32 as CubeType>::ExpandType;
    fn __expand_to_linear_pos_checked_method(
        &self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> <(u32, bool) as CubeType>::ExpandType;
    fn __expand_shape_method(&self, scope: &mut Scope) -> C::ExpandType;
}

impl<L: Layout> VirtualLayoutOperations<L::Coordinates> for L {
    fn to_linear_pos(&self, pos: L::Coordinates) -> u32 {
        L::to_linear_pos(self, pos)
    }

    fn to_linear_pos_checked(&self, pos: L::Coordinates) -> (u32, bool) {
        L::to_linear_pos_checked(self, pos)
    }

    fn shape(&self) -> L::Coordinates {
        L::shape(self)
    }
}
