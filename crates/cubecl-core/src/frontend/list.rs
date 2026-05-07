use core::marker::PhantomData;

use super::CubeType;
use crate as cubecl;
use crate::{prelude::*, unexpanded};
use cubecl_ir::{Scope, VectorSize};
use derive_more::{Deref, DerefMut};

/// Hack to avoid reimplementing `SliceIndexExpand` for every container. Rust uses `SliceIndex<[T]>`
/// for this same purpose, but we need a slightly hackier version since we can't just deref to slice.
#[repr(transparent)]
#[derive(Deref, DerefMut)]
pub struct ListExpandMarker<E: CubePrimitive, T: ListExpand<E>> {
    #[deref]
    #[deref_mut]
    list: T,
    _e: PhantomData<E>,
}

impl<E: CubePrimitive, T: ListExpand<E>> From<T> for ListExpandMarker<E, T> {
    fn from(value: T) -> Self {
        ListExpandMarker {
            list: value,
            _e: PhantomData,
        }
    }
}

impl<E: CubePrimitive, T: ListExpand<E>> ListExpandMarker<E, T> {
    pub fn from_ref(value: &T) -> &ListExpandMarker<E, T> {
        unsafe { core::mem::transmute(value) }
    }

    pub fn from_mut(value: &mut T) -> &mut ListExpandMarker<E, T> {
        unsafe { core::mem::transmute(value) }
    }
}

/// Type from which we can read values in cube functions.
/// For a mutable version, see [`ListMut`].
#[allow(clippy::len_without_is_empty)]
#[cube(expand_base_traits = "SliceOperatorExpand<T>
    + IndexExpand<NativeExpand<usize>, Output = NativeExpand<T>>
    + IndexMutExpand<NativeExpand<usize>, Output = NativeExpand<T>>")]
pub trait List<T: CubePrimitive>:
    SliceOperator<T> + CubeIndex<usize, Output = T> + CubeIndexMut<usize, Output = T> + Vectorized
{
    fn len(&self) -> usize {
        unexpanded!();
    }
}

pub trait Vectorized: CubeType<ExpandType: VectorizedExpand> {
    fn vector_size(&self) -> VectorSize {
        unexpanded!()
    }
    fn __expand_vector_size(_scope: &Scope, this: Self::ExpandType) -> VectorSize {
        this.vector_size()
    }
}

pub trait VectorizedExpand {
    fn vector_size(&self) -> VectorSize;
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        self.vector_size()
    }
}
