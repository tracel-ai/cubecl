use core::marker::PhantomData;

use crate::{
    self as cubecl,
    prelude::{Lined, LinedExpand},
    unexpanded,
};
use cubecl_ir::{Marker, VariableKind};
use cubecl_macros::{cube, intrinsic};

use crate::{
    frontend::{CubePrimitive, CubeType, ExpandElementTyped, IntoMut, indexation::Index},
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

impl<T: CubePrimitive + Clone> SharedMemory<T> {
    pub fn new<S: Index>(_size: S) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn new_lined<S: Index>(_size: S, _vectorization_factor: u32) -> SharedMemory<Line<T>> {
        SharedMemory { _val: PhantomData }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 {
        unexpanded!()
    }

    pub fn buffer_len(&self) -> u32 {
        unexpanded!()
    }

    pub fn __expand_new_lined(
        scope: &mut Scope,
        size: ExpandElementTyped<u32>,
        line_size: u32,
    ) -> <SharedMemory<Line<T>> as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var =
            scope.create_shared_array(Type::new(T::as_type(scope)).line(line_size), size, None);
        ExpandElementTyped::new(var)
    }

    pub fn vectorized<S: Index>(_size: S, _vectorization_factor: u32) -> Self {
        SharedMemory { _val: PhantomData }
    }

    pub fn __expand_vectorized(
        scope: &mut Scope,
        size: ExpandElementTyped<u32>,
        line_size: u32,
    ) -> <Self as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var =
            scope.create_shared_array(Type::new(T::as_type(scope)).line(line_size), size, None);
        ExpandElementTyped::new(var)
    }

    pub fn __expand_new(
        scope: &mut Scope,
        size: ExpandElementTyped<u32>,
    ) -> <Self as CubeType>::ExpandType {
        let size = size
            .constant()
            .expect("Shared memory need constant initialization value")
            .as_u32();
        let var = scope.create_shared_array(Type::new(T::as_type(scope)), size, None);
        ExpandElementTyped::new(var)
    }

    pub fn __expand_len(
        scope: &mut Scope,
        this: ExpandElementTyped<Self>,
    ) -> ExpandElementTyped<u32> {
        this.__expand_len_method(scope)
    }

    pub fn __expand_buffer_len(
        scope: &mut Scope,
        this: ExpandElementTyped<Self>,
    ) -> ExpandElementTyped<u32> {
        this.__expand_buffer_len_method(scope)
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
    pub fn new_lined(#[comptime] line_size: u32) -> SharedMemory<Line<T>> {
        intrinsic!(|scope| {
            let var = scope.create_shared(Type::new(T::as_type(scope)).line(line_size));
            ExpandElementTyped::new(var)
        })
    }
}

impl<T: CubePrimitive> ExpandElementTyped<SharedMemory<T>> {
    pub fn __expand_len_method(self, _scope: &mut Scope) -> ExpandElementTyped<u32> {
        len_static(&self)
    }

    pub fn __expand_buffer_len_method(self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        self.__expand_len_method(scope)
    }
}

#[cube]
impl<T: CubePrimitive + Clone> SharedMemory<T> {
    #[allow(unused_variables)]
    pub fn new_aligned(
        #[comptime] size: u32,
        #[comptime] line_size: u32,
        #[comptime] alignment: u32,
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
) -> ExpandElementTyped<u32> {
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
        pub unsafe fn index_unchecked(&self, i: u32) -> &E {
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
        pub unsafe fn index_assign_unchecked(&mut self, i: u32, value: E) {
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
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, this, idx)
    }
}

impl<T: CubePrimitive> ListExpand<T> for ExpandElementTyped<SharedMemory<T>> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index::expand(scope, self.clone(), idx)
    }
    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<T> {
        index_unchecked::expand(scope, self.clone(), idx)
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        Self::__expand_len_method(self.clone(), scope)
    }
}

impl<T: CubePrimitive> Lined for SharedMemory<T> {}
impl<T: CubePrimitive> LinedExpand for ExpandElementTyped<SharedMemory<T>> {
    fn line_size(&self) -> u32 {
        self.expand.ty.line_size()
    }
}

impl<T: CubePrimitive> ListMut<T> for SharedMemory<T> {
    fn __expand_write(
        scope: &mut Scope,
        this: ExpandElementTyped<SharedMemory<T>>,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, this, idx, value);
    }
}

impl<T: CubePrimitive> ListMutExpand<T> for ExpandElementTyped<SharedMemory<T>> {
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        idx: ExpandElementTyped<u32>,
        value: ExpandElementTyped<T>,
    ) {
        index_assign::expand(scope, self.clone(), idx, value);
    }
}
