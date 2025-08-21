use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, unexpanded};
use cubecl_std::tensor::r#virtual::{Read, VirtualTensor};

use crate::components::layout::{Coordinates, Layout};

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
    #[allow(unused)]
    pub fn to_linear_pos(&self, pos: C) -> u32 {
        intrinsic!(|scope| { self.state.__expand_to_linear_pos_method(scope, pos) })
    }

    #[allow(unused)]
    pub fn to_linear_pos_checked(&self, pos: C) -> (u32, bool) {
        intrinsic!(|scope| { self.state.__expand_to_linear_pos_checked_method(scope, pos) })
    }

    pub fn shape(&self) -> C {
        intrinsic!(|scope| { self.state.__expand_shape_method(scope) })
    }
}

impl<C: Coordinates> VirtualLayout<C> {
    pub fn new<L>(_layout: L) -> VirtualLayout<C>
    where
        L: VirtualLayoutOperations<C> + CubeType,
        L::ExpandType: VirtualLayoutOperationsExpand<C>,
    {
        unexpanded!()
    }

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

#[derive(CubeType, Clone)]
pub struct VirtualTensorView<E: Numeric, C: Coordinates, IO: Clone = Read> {
    pub tensor: VirtualTensor<E, IO>,
    pub layout: VirtualLayout<C>,
}

impl<E: Numeric, C: Coordinates, IO: Clone> Copy for VirtualTensorView<E, C, IO> {}

#[cube]
impl<E: Numeric, C: Coordinates, IO: Clone> VirtualTensorView<E, C, IO> {
    pub fn new(tensor: VirtualTensor<E, IO>, layout: VirtualLayout<C>) -> Self {
        VirtualTensorView::<E, C, IO> { tensor, layout }
    }

    #[allow(unused)]
    pub fn to_linear_pos(&self, pos: C) -> u32 {
        self.layout.to_linear_pos(pos)
    }

    #[allow(unused)]
    pub fn to_linear_pos_checked(&self, pos: C) -> (u32, bool) {
        self.layout.to_linear_pos_checked(pos)
    }

    pub fn shape(&self) -> C {
        self.layout.shape()
    }
}

pub trait VirtualLayoutOperations<C: Coordinates> {
    fn to_linear_pos(&self, pos: C) -> u32;
    fn to_linear_pos_checked(&self, pos: C) -> (u32, bool);
    fn shape(&self) -> C;
}

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
