use core::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use crate::{
    self as cubecl,
    prelude::{Lined, LinedExpand},
    unexpanded,
};
use cubecl_ir::{LineSize, Marker, VariableKind};
use cubecl_macros::{cube, intrinsic};

use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementTyped, IntoMut},
    ir::{Scope, Type},
    prelude::{
        Line, List, ListExpand, ListMut, ListMutExpand, index, index_assign, index_unchecked,
    },
};

pub type SharedMemoryExpand<T> = ExpandElementTyped<SharedMemory<T>>;
pub type SharedExpand<T> = ExpandElementTyped<Shared<T>>;

#[derive(Clone, Copy)]
pub struct Shared<E: CubePrimitive> {
    _val: PhantomData<E>,
}

#[derive(Clone, Copy)]
pub struct SharedMemory<E: CubePrimitive> {
    _val: PhantomData<E>,
}

impl<T: CubePrimitive> IntoMut for ExpandElementTyped<SharedMemory<T>> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for SharedMemory<T> {
    type ExpandType = ExpandElementTyped<SharedMemory<T>>;
}

impl<T: CubePrimitive> IntoMut for ExpandElementTyped<Shared<T>> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<T: CubePrimitive> CubeType for Shared<T> {
    type ExpandType = ExpandElementTyped<Shared<T>>;
}

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new(#[comptime] size: usize) -> Self {
        intrinsic!(|scope| {
            scope
                .create_shared_array(Type::new(T::as_type(scope)), size, None)
                .into()
        })
    }

    #[allow(unused_variables)]
    pub fn new_lined(
        #[comptime] size: usize,
        #[comptime] line_size: LineSize,
    ) -> SharedMemory<Line<T>> {
        intrinsic!(|scope| {
            scope
                .create_shared_array(Type::new(T::as_type(scope)).line(line_size), size, None)
                .into()
        })
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        intrinsic!(|_| len_static(&self))
    }

    pub fn buffer_len(&self) -> usize {
        self.len()
    }
}

#[cube]
impl<T: CubePrimitive> Shared<T> {
    pub fn new() -> Self {
        intrinsic!(|scope| {
            let var = scope.create_shared(Type::new(T::as_type(scope)));
            ExpandElementTyped::new(var)
        })
    }
}

pub trait AsRefExpand<T: CubeType> {
    /// Converts this type into a shared reference of the (usually inferred) input type.
    fn __expand_as_ref_method(self, scope: &mut Scope) -> T::ExpandType;
}
impl<T: CubePrimitive> AsRefExpand<T> for ExpandElementTyped<T> {
    fn __expand_as_ref_method(self, _scope: &mut Scope) -> ExpandElementTyped<T> {
        self
    }
}
pub trait AsMutExpand<T: CubeType> {
    /// Converts this type into a shared reference of the (usually inferred) input type.
    fn __expand_as_mut_method(self, scope: &mut Scope) -> T::ExpandType;
}
impl<T: CubePrimitive> AsMutExpand<T> for ExpandElementTyped<T> {
    fn __expand_as_mut_method(self, _scope: &mut Scope) -> <T as CubeType>::ExpandType {
        self
    }
}

/// Type inference won't allow things like assign to work normally, so we need to manually call
/// `as_ref` or `as_mut` for those. Things like barrier ops should take `AsRef` so the conversion
/// is automatic.
impl<T: CubePrimitive> AsRef<T> for Shared<T> {
    fn as_ref(&self) -> &T {
        unexpanded!()
    }
}
impl<T: CubePrimitive> AsRefExpand<T> for SharedExpand<T> {
    fn __expand_as_ref_method(self, _scope: &mut Scope) -> <T as CubeType>::ExpandType {
        self.expand.into()
    }
}

impl<T: CubePrimitive> AsMut<T> for Shared<T> {
    fn as_mut(&mut self) -> &mut T {
        unexpanded!()
    }
}
impl<T: CubePrimitive> AsMutExpand<T> for SharedExpand<T> {
    fn __expand_as_mut_method(self, _scope: &mut Scope) -> <T as CubeType>::ExpandType {
        self.expand.into()
    }
}

impl<T: CubePrimitive> Default for Shared<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: CubePrimitive> Shared<T> {
    pub fn __expand_default(scope: &mut Scope) -> <Self as CubeType>::ExpandType {
        Self::__expand_new(scope)
    }
}

#[cube]
impl<T: CubePrimitive> Shared<Line<T>> {
    #[allow(unused_variables)]
    pub fn new_lined(#[comptime] line_size: LineSize) -> SharedMemory<Line<T>> {
        intrinsic!(|scope| {
            let var = scope.create_shared(Type::new(T::as_type(scope)).line(line_size));
            ExpandElementTyped::new(var)
        })
    }
}

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new_aligned(
        #[comptime] size: usize,
        #[comptime] line_size: LineSize,
        #[comptime] alignment: usize,
    ) -> SharedMemory<Line<T>> {
        intrinsic!(|scope| {
            let var = scope.create_shared_array(
                Type::new(T::as_type(scope)).line(line_size),
                size,
                Some(alignment),
            );
            ExpandElementTyped::new(var)
        })
    }

    /// Frees the shared memory for reuse, if possible on the target runtime.
    ///
    /// # Safety
    /// *Must* be used in uniform control flow
    /// *Must not* have any dangling references to this shared memory
    pub unsafe fn free(self) {
        intrinsic!(|scope| { scope.register(Marker::Free(*self.expand)) })
    }
}

fn len_static<T: CubePrimitive>(
    shared: &ExpandElementTyped<SharedMemory<T>>,
) -> ExpandElementTyped<usize> {
    let VariableKind::SharedArray { length, .. } = shared.expand.kind else {
        unreachable!("Kind of shared memory is always shared memory")
    };
    length.into()
}

/// Module that contains the implementation details of the index functions.
mod indexation {
    use cubecl_ir::{IndexAssignOperator, IndexOperator, Operator};

    use crate::ir::Instruction;

    use super::*;

    type SharedMemoryExpand<E> = ExpandElementTyped<SharedMemory<E>>;

    #[cube]
    impl<E: CubePrimitive> SharedMemory<E> {
        /// Perform an unchecked index into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_unchecked(&self, i: usize) -> &E {
            intrinsic!(|scope| {
                let out = scope.create_local(self.expand.ty);
                scope.register(Instruction::new(
                    Operator::UncheckedIndex(IndexOperator {
                        list: *self.expand,
                        index: i.expand.consume(),
                        line_size: 0,
                        unroll_factor: 1,
                    }),
                    *out,
                ));
                out.into()
            })
        }

        /// Perform an unchecked index assignment into the array
        ///
        /// # Safety
        /// Out of bounds indexing causes undefined behaviour and may segfault. Ensure index is
        /// always in bounds
        #[allow(unused_variables)]
        pub unsafe fn index_assign_unchecked(&mut self, i: usize, value: E) {
            intrinsic!(|scope| {
                scope.register(Instruction::new(
                    Operator::UncheckedIndexAssign(IndexAssignOperator {
                        index: i.expand.consume(),
                        value: value.expand.consume(),
                        line_size: 0,
                        unroll_factor: 1,
                    }),
                    *self.expand,
                ));
            })
        }
    }
}

impl<T: CubePrimitive> List<T> for SharedMemory<T> {
    fn __expand_read(
        scope: &mut Scope,
        this: ExpandElementTyped<SharedMemory<T>>,
        idx: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, this, idx)
    }
}

impl<T: CubePrimitive> Deref for SharedMemory<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> DerefMut for SharedMemory<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unexpanded!()
    }
}

impl<T: CubePrimitive> ListExpand<T> for ExpandElementTyped<SharedMemory<T>> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<usize>,
    ) -> ExpandElementTyped<T> {
        index_unchecked::expand(scope, self.clone(), idx)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<usize> {
        Self::__expand_len_method(self.clone(), scope)
    }
}

impl<T: CubePrimitive> Lined for SharedMemory<T> {}
impl<T: CubePrimitive> LinedExpand for ExpandElementTyped<SharedMemory<T>> {
    fn line_size(&self) -> LineSize {
        self.expand.ty.line_size()
    }
}

impl<T: CubePrimitive> ListMut<T> for SharedMemory<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: ExpandElementTyped<SharedMemory<T>>,
        idx: ExpandElementTyped<usize>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for ExpandElementTyped<SharedMemory<T>> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<usize>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
