use std::marker::PhantomData;

use crate::{self as cubecl, unexpanded};
use cubecl::prelude::*;
use cubecl_ir::{Branch, ExpandElement, Item, RangeLoop, Variable};

pub struct ReadOnly;
pub struct ReadWrite;

/// A read-only contiguous list of elements
///
/// # Safety
///
/// Since data can't be deallocated during kernel execution, this is safe.
#[derive(Clone, Copy)]
pub struct SliceV2<E: CubePrimitive, IO: SliceVisibility> {
    _e: PhantomData<E>,
    _io: PhantomData<IO>,
    _offset: PhantomData<u32>,
    length: u32,
}

#[derive(CubeType)]
pub enum SliceOrigin<E: CubePrimitive> {
    Tensor(Tensor<E>),
    Array(Array<E>),
    SharedMemory(SharedMemory<E>),
}

impl<E: CubePrimitive, IO: SliceVisibility> Iterator for SliceV2<E, IO> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

pub trait SliceVisibility {}

impl SliceVisibility for ReadOnly {}

impl SliceVisibility for ReadWrite {}

pub struct SliceV2Expand<E: CubePrimitive, IO: SliceVisibility> {
    pub(crate) origin: SliceOriginExpand<E>,
    pub(crate) io: PhantomData<IO>,
    pub(crate) offset: ExpandElementTyped<u32>,
    pub(crate) length: ExpandElementTyped<u32>,
}

impl<E: CubePrimitive, IO: SliceVisibility> SliceV2Expand<E, IO> {
    pub fn __to_raw_parts(&self) -> (Variable, Variable) {
        let expand = match self.origin.clone() {
            SliceOriginExpand::Tensor(expand) => expand.expand,
            SliceOriginExpand::Array(expand) => expand.expand,
            SliceOriginExpand::SharedMemory(expand) => expand.expand,
        };

        (*expand, *self.offset.expand)
    }
}


#[cube]
impl<E: CubePrimitive, IO: SliceVisibility> SliceV2<E, IO> {
    pub fn into_lined(_origin: SliceOrigin<E>, _offset: u32, _length: u32) -> Self {
        intrinsic!(|scope| {
        })
    }
}


impl<E: CubePrimitive, IO: SliceVisibility> SliceV2<E, IO> {
    pub fn new(_origin: SliceOrigin<E>, _offset: u32, _length: u32) -> Self {
        unexpanded!()
    }
    pub fn __expand_new(
        scope: &mut Scope,
        origin: SliceOriginExpand<E>,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, IO> {
        Self::__expand_new_expand(scope, origin, start, end)
    }
    pub fn __expand_new_expand(
        scope: &mut Scope,
        origin: SliceOriginExpand<E>,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceV2Expand<E, IO> {
        let length = cubecl::frontend::sub::expand(scope, end.into(), start.clone().into());

        SliceV2Expand::<E, IO> {
            origin,
            io: PhantomData,
            offset: start,
            length,
        }
    }
}

#[cube]
impl<E: CubePrimitive, IO: SliceVisibility> SliceV2<E, IO> {
    /// Get the length of the slice.
    pub fn len(&self) -> u32 {
        self.length
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeType for SliceV2<E, IO> {
    type ExpandType = SliceV2Expand<E, IO>;
}

impl<E: CubePrimitive, IO: SliceVisibility> Init for SliceV2Expand<E, IO> {
    fn init(self, _scope: &mut cubecl_ir::Scope) -> Self {
        self
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeDebug for SliceV2Expand<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> Clone for SliceV2Expand<E, IO> {
    fn clone(&self) -> Self {
        Self {
            origin: self.origin.clone(),
            offset: self.offset.clone(),
            length: self.length.clone(),
            io: PhantomData,
        }
    }
}

// TODO: Fix
impl<E: CubePrimitive> SizedContainer for SliceV2<E, ReadOnly> {
    type Item = E;
}

impl<E: CubePrimitive> Iterable<E> for SliceV2Expand<E, ReadOnly> {
    fn expand(
        self,
        scope: &mut Scope,
        mut body: impl FnMut(&mut Scope, <E as CubeType>::ExpandType),
    ) {
        let index_ty = Item::new(u32::as_elem(scope));
        let len: ExpandElement = self.length.clone().into();

        let mut child = scope.child();
        let i = child.create_local_restricted(index_ty);

        let index = i.clone().into();
        let item = index::expand(&mut child, self, index);
        body(&mut child, item);

        scope.register(Branch::RangeLoop(Box::new(RangeLoop {
            i: *i,
            start: 0u32.into(),
            end: *len,
            step: None,
            inclusive: false,
            scope: child,
        })));
    }

    fn expand_unroll(
        self,
        _scope: &mut Scope,
        _body: impl FnMut(&mut Scope, <E as CubeType>::ExpandType),
    ) {
        unimplemented!("Can't unroll slice iterator")
    }
}
impl<E: CubePrimitive> CubeIndex for SliceV2<E, ReadOnly> {
    type Output = E;

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.__expand_read_method(scope, index)
    }
}

impl<E: CubePrimitive> CubeIndexExpand for SliceV2Expand<E, ReadOnly> {
    type Output = E::ExpandType;

    fn expand_index(self, scope: &mut Scope, index: ExpandElementTyped<u32>) -> Self::Output {
        self.__expand_read_method(scope, index)
    }
    fn expand_index_unchecked(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> Self::Output {
        self.__expand_read_unchecked_method(scope, index)
    }
}

impl<E: CubePrimitive> List<E> for SliceV2<E, ReadOnly> {}
impl<E: CubePrimitive> ListExpand<E> for SliceV2Expand<E, ReadOnly> {
    fn __expand_read_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(scope, self.origin.clone(), self.offset.clone(), index)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset_unchecked::expand::<E>(scope, self.origin.clone(), self.offset.clone(), index)
    }
}

impl<E: CubePrimitive> CubeIndex for SliceV2<E, ReadWrite> {
    type Output = E;

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.__expand_read_method(scope, index)
    }
}

impl<E: CubePrimitive> CubeIndexExpand for SliceV2Expand<E, ReadWrite> {
    type Output = E::ExpandType;

    fn expand_index(self, scope: &mut Scope, index: ExpandElementTyped<u32>) -> Self::Output {
        self.__expand_read_method(scope, index)
    }
    fn expand_index_unchecked(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> Self::Output {
        self.__expand_read_unchecked_method(scope, index)
    }
}

impl<E: CubePrimitive> List<E> for SliceV2<E, ReadWrite> {}
impl<E: CubePrimitive> ListExpand<E> for SliceV2Expand<E, ReadWrite> {
    fn __expand_read_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(scope, self.origin.clone(), self.offset.clone(), index)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset_unchecked::expand::<E>(scope, self.origin.clone(), self.offset.clone(), index)
    }
}

impl<E: CubePrimitive> CubeIndexMut for SliceV2<E, ReadWrite> {
    fn expand_index_mut(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<E>,
    ) {
        array.__expand_write_method(scope, index, value)
    }
}

impl<E: CubePrimitive> CubeIndexMutExpand for SliceV2Expand<E, ReadWrite> {
    type Output = E::ExpandType;

    fn expand_index_mut(
        self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: Self::Output,
    ) {
        self.__expand_write_method(scope, index, value)
    }
}

impl<E: CubePrimitive> ListMut<E> for SliceV2<E, ReadWrite> {}
impl<E: CubePrimitive> ListMutExpand<E> for SliceV2Expand<E, ReadWrite> {
    fn __expand_write_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<E>,
    ) {
        write_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            value,
        )
    }
}

mod read_offset {
    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut cubecl::prelude::Scope,
        origin: SliceOriginExpand<E>,
        offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        index: <u32 as cubecl::prelude::CubeType>::ExpandType,
    ) -> <E as cubecl::prelude::CubeType>::ExpandType {
        let index = cubecl::frontend::add::expand(scope, offset.into(), index.into());

        match origin {
            SliceOriginExpand::Tensor(expand) => index::expand(scope, expand, index),
            SliceOriginExpand::Array(expand) => index::expand(scope, expand, index),
            SliceOriginExpand::SharedMemory(expand) => index::expand(scope, expand, index),
        }
    }
}

mod read_offset_unchecked {
    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut cubecl::prelude::Scope,
        origin: SliceOriginExpand<E>,
        offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        index: <u32 as cubecl::prelude::CubeType>::ExpandType,
    ) -> <E as cubecl::prelude::CubeType>::ExpandType {
        let index = cubecl::frontend::add::expand(scope, offset.into(), index.into());

        match origin {
            SliceOriginExpand::Tensor(expand) => index_unchecked::expand(scope, expand, index),
            SliceOriginExpand::Array(expand) => index_unchecked::expand(scope, expand, index),
            SliceOriginExpand::SharedMemory(expand) => {
                index_unchecked::expand(scope, expand, index)
            }
        }
    }
}

mod write_offset {
    use super::*;

    pub fn expand<E: CubePrimitive>(
        scope: &mut cubecl::prelude::Scope,
        origin: SliceOriginExpand<E>,
        offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        index: <u32 as cubecl::prelude::CubeType>::ExpandType,
        value: <E as cubecl::prelude::CubeType>::ExpandType,
    ) {
        let index = cubecl::frontend::add::expand(scope, offset.into(), index.into());

        match origin {
            SliceOriginExpand::Tensor(expand) => index_assign::expand(scope, expand, index, value),
            SliceOriginExpand::Array(expand) => index_assign::expand(scope, expand, index, value),
            SliceOriginExpand::SharedMemory(expand) => {
                index_assign::expand(scope, expand, index, value)
            }
        }
    }
}
