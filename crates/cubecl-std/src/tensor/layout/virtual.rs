use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, unexpanded};

use crate::tensor::layout::{Coordinates, Layout, LayoutExpand};

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
        L: Layout<Coordinates = C, SourceCoordinates = S> + CubeType,
        L::ExpandType: LayoutExpand<Coordinates = C, SourceCoordinates = S>,
    {
        unexpanded!()
    }

    /// Expand function of [VirtualLayout::__expand_new]
    pub fn __expand_new<L>(_scope: &mut Scope, layout: L::ExpandType) -> VirtualLayoutExpand<C, S>
    where
        L: Layout<Coordinates = C, SourceCoordinates = S> + CubeType,
        L::ExpandType: LayoutExpand<Coordinates = C, SourceCoordinates = S> + 'static,
    {
        VirtualLayoutExpand::new::<L>(layout)
    }
}

impl<C: Coordinates, S: Coordinates> VirtualLayoutExpand<C, S> {
    /// Create a new virtual layout from a concrete one
    pub fn new<L>(layout: L::ExpandType) -> VirtualLayoutExpand<C, S>
    where
        L: Layout<Coordinates = C, SourceCoordinates = S> + CubeType,
        L::ExpandType: LayoutExpand<Coordinates = C, SourceCoordinates = S> + 'static,
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

pub trait VirtualLayoutOperationsExpand<C: CubeType, S: CubeType> {
    fn __expand_to_source_pos_method(
        &self,
        scope: &mut Scope,
        pos: <C as CubeType>::ExpandType,
    ) -> <S as CubeType>::ExpandType;
    fn __expand_to_source_pos_checked_method(
        &self,
        scope: &mut Scope,
        pos: <C as CubeType>::ExpandType,
    ) -> <(S, bool) as CubeType>::ExpandType;
    fn __expand_shape_method(&self, scope: &mut Scope) -> <C as CubeType>::ExpandType;
    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: <C as CubeType>::ExpandType,
    ) -> ExpandElementTyped<bool>;
}

impl<L: LayoutExpand> VirtualLayoutOperationsExpand<L::Coordinates, L::SourceCoordinates> for L {
    fn __expand_to_source_pos_method(
        &self,
        scope: &mut Scope,
        pos: <L::Coordinates as CubeType>::ExpandType,
    ) -> <L::SourceCoordinates as CubeType>::ExpandType {
        <L as LayoutExpand>::__expand_to_source_pos_method(self.clone(), scope, pos)
    }

    fn __expand_to_source_pos_checked_method(
        &self,
        scope: &mut Scope,
        pos: <L::Coordinates as CubeType>::ExpandType,
    ) -> <(L::SourceCoordinates, bool) as CubeType>::ExpandType {
        <L as LayoutExpand>::__expand_to_source_pos_checked_method(self.clone(), scope, pos)
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> <L::Coordinates as CubeType>::ExpandType {
        <L as LayoutExpand>::__expand_shape_method(self.clone(), scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: <L::Coordinates as CubeType>::ExpandType,
    ) -> ExpandElementTyped<bool> {
        <L as LayoutExpand>::__expand_is_in_bounds_method(self.clone(), scope, pos)
    }
}
