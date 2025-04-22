use std::{marker::PhantomData, sync::Arc};

use crate as cubecl;
use cubecl::prelude::*;

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
    offset: PhantomData<u32>,
    len: PhantomData<u32>,
}

pub trait SliceVisibility {
    type ListType<E>: Clone;
}

impl SliceVisibility for ReadOnly {
    type ListType<E> = Arc<dyn ListExpand<E>>;
}

impl SliceVisibility for ReadWrite {
    type ListType<E> = Arc<dyn ListMutExpand<E>>;
}

pub struct SliceV2Expand<E, IO: SliceVisibility> {
    list: IO::ListType<E>,
    offset: ExpandElementTyped<u32>,
    len: ExpandElementTyped<u32>,
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
            list: self.list.clone(),
            offset: self.offset.clone(),
            len: self.len.clone(),
        }
    }
}

impl<E: CubePrimitive> List<E> for SliceV2<E, ReadOnly> {}

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

    fn expand_index(
        scope: &mut Scope,
        array: Self,
        index: ExpandElementTyped<u32>,
    ) -> Self::Output {
        array.__expand_read_method(scope, index)
    }
}

impl<E: CubePrimitive> ListExpand<E> for SliceV2Expand<E, ReadOnly> {
    fn __expand_read_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(scope, self.list.as_ref(), self.offset.clone(), index)
    }
}

impl<E: CubePrimitive> List<E> for SliceV2<E, ReadWrite> {}

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

    fn expand_index(
        scope: &mut Scope,
        array: Self,
        index: ExpandElementTyped<u32>,
    ) -> Self::Output {
        array.__expand_read_method(scope, index)
    }
}

impl<E: CubePrimitive> ListExpand<E> for SliceV2Expand<E, ReadWrite> {
    fn __expand_read_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(scope, self.list.as_ref(), self.offset.clone(), index)
    }
}

mod read_offset {
    use super::*;

    pub fn expand<E: CubePrimitive>(
        context: &mut cubecl::prelude::Scope,
        list: &dyn ListExpand<E>,
        offset: <u32 as cubecl::prelude::CubeType>::ExpandType,
        index: <u32 as cubecl::prelude::CubeType>::ExpandType,
    ) -> <E as cubecl::prelude::CubeType>::ExpandType {
        let position = cubecl::frontend::add::expand(context, offset.into(), index.into());
        list.__expand_read_method(context, position.into())
    }
}
