use std::{marker::PhantomData, sync::Arc};

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic, io::read_masked, unexpanded};

use crate::tensor::{
    layout::{Coordinates, Layout},
    r#virtual::{Read, ReadWrite},
};

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

#[derive(Clone)]
pub struct TensorView<E: Numeric, C: Coordinates, IO: Clone = Read> {
    pub layout: VirtualLayout<C>,
    _list: PhantomData<(E, IO)>,
}

impl<E: Numeric, C: Coordinates, IO: Clone> Copy for TensorView<E, C, IO> {}

#[derive(Clone)]
enum ListType<E: Numeric> {
    Read(Arc<dyn ListExpand<Line<E>>>),
    ReadWrite(Arc<dyn ListMutExpand<Line<E>>>),
}

impl<E: Numeric> ListType<E> {
    pub fn read(&self) -> &dyn ListExpand<Line<E>> {
        match self {
            ListType::Read(list) => &**list,
            ListType::ReadWrite(list) => &**list,
        }
    }

    pub fn write(&self) -> &dyn ListMutExpand<Line<E>> {
        match self {
            ListType::Read(_) => panic!("Can't write to readonly list"),
            ListType::ReadWrite(list) => &**list,
        }
    }
}

#[derive(Clone)]
pub struct TensorViewExpand<E: Numeric, C: Coordinates, IO: Clone = Read> {
    tensor: ListType<E>,
    layout: VirtualLayoutExpand<C>,
    _io: PhantomData<IO>,
}

impl<E: Numeric, C: Coordinates, IO: Clone> CubeType for TensorView<E, C, IO> {
    type ExpandType = TensorViewExpand<E, C, IO>;
}

impl<E: Numeric, C: Coordinates, IO: Clone> IntoMut for TensorViewExpand<E, C, IO> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<E: Numeric, C: Coordinates, IO: Clone> CubeDebug for TensorViewExpand<E, C, IO> {}

impl<E: Numeric, C: Coordinates> TensorView<E, C, Read> {
    #[allow(unused_variables)]
    pub fn new<T>(tensor: T, layout: VirtualLayout<C>) -> Self
    where
        T: List<Line<E>> + CubeType,
        T::ExpandType: ListExpand<Line<E>> + 'static,
    {
        TensorView {
            layout,
            _list: PhantomData,
        }
    }

    pub fn __expand_new<T>(
        _scope: &mut Scope,
        tensor: T::ExpandType,
        layout: VirtualLayoutExpand<C>,
    ) -> TensorViewExpand<E, C, Read>
    where
        T: List<Line<E>> + CubeType,
        T::ExpandType: ListExpand<Line<E>> + 'static,
    {
        TensorViewExpand::<E, C, Read> {
            tensor: ListType::Read(Arc::new(tensor)),
            layout,
            _io: PhantomData,
        }
    }
}

impl<E: Numeric, C: Coordinates> TensorView<E, C, ReadWrite> {
    pub fn new_mut<T>(_tensor: T, _layout: VirtualLayout<C>) -> TensorView<E, C, ReadWrite>
    where
        T: ListMut<Line<E>> + CubeType,
        T::ExpandType: ListMutExpand<Line<E>> + 'static,
    {
        unexpanded!()
    }

    pub fn __expand_new_mut<T>(
        _scope: &mut Scope,
        tensor: T::ExpandType,
        layout: VirtualLayoutExpand<C>,
    ) -> TensorViewExpand<E, C, ReadWrite>
    where
        T: ListMut<Line<E>> + CubeType,
        T::ExpandType: ListMutExpand<Line<E>> + 'static,
    {
        TensorViewExpand::<E, C, ReadWrite> {
            tensor: ListType::ReadWrite(Arc::new(tensor)),
            layout,
            _io: PhantomData,
        }
    }
}

#[cube]
impl<E: Numeric, C: Coordinates, IO: Clone> TensorView<E, C, IO> {
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

#[allow(unused_variables)]
impl<E: Numeric, C: Coordinates, IO: Clone> TensorView<E, C, IO> {
    pub fn read(&self, pos: C) -> Line<E> {
        unexpanded!()
    }

    pub fn read_checked(&self, pos: C) -> Line<E> {
        unexpanded!()
    }

    pub fn slice(&self, pos: C, size: u32) -> Slice<Line<E>, ReadOnly> {
        unexpanded!()
    }
}

impl<E: Numeric, C: Coordinates, IO: Clone> TensorViewExpand<E, C, IO> {
    pub fn __expand_read_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<Line<E>> {
        let read_pos = self.clone().__expand_to_linear_pos_method(scope, pos);
        self.tensor.read().__expand_read_method(scope, read_pos)
    }

    pub fn __expand_read_checked_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<Line<E>> {
        let slice = self.clone().tensor.read().__expand_to_slice_method(scope);
        let (read_pos, in_bounds) = self.__expand_to_linear_pos_checked_method(scope, pos);
        let zero = Line::__expand_cast_from(scope, 0.into());
        read_masked::expand::<Line<E>>(scope, in_bounds, slice, read_pos, zero)
    }

    pub fn __expand_slice_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
        size: ExpandElementTyped<u32>,
    ) -> SliceExpand<Line<E>, ReadOnly> {
        let start = self.clone().__expand_to_linear_pos_method(scope, pos);
        let end = add::expand(scope, start.clone(), size);
        self.tensor.read().__expand_slice_method(scope, start, end)
    }
}

#[allow(unused_variables)]
impl<E: Numeric, C: Coordinates> TensorView<E, C, ReadWrite> {
    pub fn write(&self, pos: C, value: Line<E>) {
        unexpanded!()
    }

    pub fn write_checked(&self, pos: C, value: Line<E>) {
        unexpanded!()
    }
}

impl<E: Numeric, C: Coordinates> TensorViewExpand<E, C, ReadWrite> {
    pub fn __expand_write_method(
        self,
        scope: &mut Scope,
        pos: C::ExpandType,
        value: ExpandElementTyped<Line<E>>,
    ) {
        let write_pos = self.clone().__expand_to_linear_pos_method(scope, pos);
        self.tensor
            .write()
            .__expand_write_method(scope, write_pos, value)
    }

    pub fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: C::ExpandType,
        value: ExpandElementTyped<Line<E>>,
    ) {
        let (write_pos, in_bounds) = self
            .clone()
            .__expand_to_linear_pos_checked_method(scope, pos);
        if_expand(scope, in_bounds.into(), |scope| {
            self.tensor
                .write()
                .__expand_write_method(scope, write_pos, value)
        });
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
