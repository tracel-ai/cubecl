use std::{marker::PhantomData, num::NonZero};

use crate::{self as cubecl, unexpanded};
use cubecl::prelude::*;
use cubecl_ir::{Branch, Elem, ExpandElement, FloatKind, Item, RangeLoop, Variable};
use cubecl_macros::intrinsic;

#[derive(Clone)]
pub struct ReadOnly;
#[derive(Clone)]
pub struct ReadWrite;

/// A read-only contiguous list of elements
///
/// # Safety
///
/// Since data can't be deallocated during kernel execution, this is safe.
#[derive(Clone, Copy)]
pub struct Slice<E: CubePrimitive, IO: SliceVisibility = ReadOnly> {
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

impl<E: CubePrimitive, IO: SliceVisibility> Iterator for Slice<E, IO> {
    type Item = E;

    fn next(&mut self) -> Option<Self::Item> {
        unexpanded!()
    }
}

pub trait SliceVisibility {}

impl SliceVisibility for ReadOnly {}

impl SliceVisibility for ReadWrite {}

pub struct SliceExpand<E: CubePrimitive, IO: SliceVisibility> {
    pub(crate) origin: SliceOriginExpand<E>,
    pub(crate) io: PhantomData<IO>,
    pub(crate) offset: ExpandElementTyped<u32>,
    pub(crate) length: ExpandElementTyped<u32>,
    pub(crate) line_size: Option<u32>,
}

impl<E: CubePrimitive, IO: SliceVisibility> SliceExpand<E, IO> {
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
impl<E: CubePrimitive, IO: SliceVisibility> Slice<Line<E>, IO> {
    /// Reinterprets how items are loaded and stored in memory.slicebase
    ///
    /// # Warning
    ///
    /// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
    #[allow(unused_variables)]
    pub fn with_line_size(&self, #[comptime] line_size: u32) -> Slice<Line<E>, IO> {
        intrinsic!(|scope| {
            let (input, offset) = self.__to_raw_parts();
            let mut item = input.item;

            if line_size as u8 == item.vectorization.unwrap_or(NonZero::new(1).unwrap()).get() {
                return self;
            }

            let current = input.item.vectorization.map(|a| a.get()).unwrap_or(1) as u32;
            let mut out = self.clone();

            if current < line_size {
                let ratio = line_size / current;
                let length = cubecl::frontend::div::expand(scope, self.length, ratio.into());
                let offset = cubecl::frontend::div::expand(scope, self.offset, ratio.into());
                out.length = length;
                out.offset = offset;
            } else {
                let ratio = current / line_size;
                let length = cubecl::frontend::mul::expand(scope, self.length, ratio.into());
                let offset = cubecl::frontend::mul::expand(scope, self.offset, ratio.into());
                out.length = length;
                out.offset = offset;
            }

            out.line_size = Some(line_size);
            out
        })
    }
}

#[cube]
impl<E: CubePrimitive, IO: SliceVisibility> Slice<E, IO> {
    /// Returns the same slice, but with lines of length 1.
    pub fn into_lined(&self) -> Slice<Line<E>, IO> {
        intrinsic!(|_scope| {
            SliceExpand::<Line<E>, IO> {
                origin: self.origin.cast_unchecked(),
                io: self.io.clone(),
                offset: self.offset.clone(),
                length: self.length.clone(),
                line_size: None,
            }
        })
    }
    /// Returns the same slice, but with lines of length 1.
    /// Try to cast the slice to the given type and panic if the type isn't the same.
    ///
    /// This function should only be used to satisfy the Rust type system, when two generic
    /// types are supposed to be the same.
    pub fn try_cast_unchecked<T: CubePrimitive>(&self) -> Slice<T, IO> {
        intrinsic!(|scope| {
            if T::as_elem(scope) != E::as_elem(scope) && !is_tf32::<E, T>(scope) {
                let elems = [T::as_elem(scope), E::as_elem(scope)];
                let is_flex32_cast = elems.contains(&Elem::Float(FloatKind::F32))
                    && elems.contains(&Elem::Float(FloatKind::Flex32));

                if !is_flex32_cast {
                    panic!(
                        "Try cast unchecked should only be used to satisfy the rust type system."
                    )
                }
            }

            SliceExpand::<T, IO> {
                origin: self.origin.cast_unchecked(),
                io: self.io.clone(),
                offset: self.offset.clone(),
                length: self.length.clone(),
                line_size: self.line_size.clone(),
            }
        })
    }
}

impl<E: CubePrimitive> SliceOriginExpand<E> {
    fn cast_unchecked<T: CubePrimitive>(self) -> SliceOriginExpand<T> {
        match self {
            SliceOriginExpand::Tensor(expand) => {
                SliceOriginExpand::<T>::Tensor(expand.expand.into())
            }
            SliceOriginExpand::Array(expand) => SliceOriginExpand::<T>::Array(expand.expand.into()),
            SliceOriginExpand::SharedMemory(expand) => {
                SliceOriginExpand::<T>::SharedMemory(expand.expand.into())
            }
        }
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> Slice<E, IO> {
    pub fn new(_origin: SliceOrigin<E>, _offset: u32, _length: u32) -> Self {
        unexpanded!()
    }
    pub fn __expand_new(
        scope: &mut Scope,
        origin: SliceOriginExpand<E>,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, IO> {
        Self::__expand_new_expand(scope, origin, start, end)
    }
    pub fn __expand_new_expand(
        scope: &mut Scope,
        origin: SliceOriginExpand<E>,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> SliceExpand<E, IO> {
        let length = cubecl::frontend::sub::expand(scope, end, start.clone());

        SliceExpand::<E, IO> {
            origin,
            io: PhantomData,
            offset: start,
            length,
            line_size: None,
        }
    }
}

#[cube]
impl<E: CubePrimitive, IO: SliceVisibility> Slice<E, IO> {
    /// Get the length of the slice.
    pub fn len(&self) -> u32 {
        self.length
    }
    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeType for Slice<E, IO> {
    type ExpandType = SliceExpand<E, IO>;
}

impl<E: CubePrimitive, IO: SliceVisibility> IntoMut for SliceExpand<E, IO> {
    fn into_mut(self, _scope: &mut cubecl_ir::Scope) -> Self {
        self
    }
}

impl<E: CubePrimitive, IO: SliceVisibility> CubeDebug for SliceExpand<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility> Clone for SliceExpand<E, IO> {
    fn clone(&self) -> Self {
        Self {
            origin: self.origin.clone(),
            offset: self.offset.clone(),
            length: self.length.clone(),
            line_size: self.line_size,
            io: PhantomData,
        }
    }
}

// TODO: Fix
impl<E: CubePrimitive> SizedContainer for Slice<E, ReadOnly> {
    type Item = E;
}

impl<E: CubePrimitive> Iterable<E> for SliceExpand<E, ReadOnly> {
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
impl<E: CubePrimitive> CubeIndex for Slice<E, ReadOnly> {
    type Output = E;

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.__expand_read_method(scope, index)
    }
}

impl<E: CubePrimitive> CubeIndexExpand for SliceExpand<E, ReadOnly> {
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

impl<E: CubePrimitive> List<E> for Slice<E, ReadOnly> {}
impl<E: CubePrimitive> ListExpand<E> for SliceExpand<E, ReadOnly> {
    fn __expand_read_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            self.line_size,
            true,
        )
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            self.line_size,
            false,
        )
    }
}

impl<E: CubePrimitive> CubeIndex for Slice<E, ReadWrite> {
    type Output = E;

    fn expand_index(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
    ) -> <Self::Output as CubeType>::ExpandType {
        array.__expand_read_method(scope, index)
    }
}

impl<E: CubePrimitive> CubeIndexExpand for SliceExpand<E, ReadWrite> {
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

impl<E: CubePrimitive> List<E> for Slice<E, ReadWrite> {}
impl<E: CubePrimitive> ListExpand<E> for SliceExpand<E, ReadWrite> {
    fn __expand_read_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            self.line_size,
            true,
        )
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut cubecl_ir::Scope,
        index: ExpandElementTyped<u32>,
    ) -> <E as CubeType>::ExpandType {
        read_offset::expand::<E>(
            scope,
            self.origin.clone(),
            self.offset.clone(),
            index,
            self.line_size,
            false,
        )
    }
}

impl<E: CubePrimitive> CubeIndexMut for Slice<E, ReadWrite> {
    fn expand_index_mut(
        scope: &mut Scope,
        array: Self::ExpandType,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<E>,
    ) {
        array.__expand_write_method(scope, index, value)
    }
}

impl<E: CubePrimitive> CubeIndexMutExpand for SliceExpand<E, ReadWrite> {
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

impl<E: CubePrimitive> ListMut<E> for Slice<E, ReadWrite> {}
impl<E: CubePrimitive> ListMutExpand<E> for SliceExpand<E, ReadWrite> {
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
            self.line_size,
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
        line_size: Option<u32>,
        checked: bool,
    ) -> <E as cubecl::prelude::CubeType>::ExpandType {
        let index = cubecl::frontend::add::expand(scope, offset, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => {
                expand_index_native::<Tensor<E>>(scope, expand, index, line_size, checked)
            }
            SliceOriginExpand::Array(expand) => {
                expand_index_native::<Array<E>>(scope, expand, index, line_size, checked)
            }
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_native::<SharedMemory<E>>(scope, expand, index, line_size, checked)
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
        line_size: Option<u32>,
    ) {
        let index = cubecl::frontend::add::expand(scope, offset, index);

        match origin {
            SliceOriginExpand::Tensor(expand) => expand_index_assign_native::<Tensor<E>>(
                scope, expand, index, value, line_size, true,
            ),
            SliceOriginExpand::Array(expand) => expand_index_assign_native::<Array<E>>(
                scope, expand, index, value, line_size, false,
            ),
            SliceOriginExpand::SharedMemory(expand) => {
                expand_index_assign_native::<SharedMemory<E>>(
                    scope, expand, index, value, line_size, false,
                )
            }
        }
    }
}
