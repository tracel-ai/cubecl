//! This module exposes barrier for asynchronous data transfer

use std::marker::PhantomData;

use cubecl_ir::ExpandElement;

use crate::{
    ir::{BarrierOps, Item, Scope},
    unexpanded,
};

use super::{
    CubeDebug, CubePrimitive, CubeType, ExpandElementTyped, Init, IntoRuntime, Line, Slice,
    SliceMut,
};

pub trait BarrierLevel: Clone + Copy {}

#[derive(Copy, Clone)]
pub struct Unit {}
#[derive(Copy, Clone)]
pub struct Cube {}

impl BarrierLevel for Unit {}
impl BarrierLevel for Cube {}

/// A mechanism for awaiting on asynchronous data transfers
/// Works at the Cube and Unit level
#[derive(Clone, Copy)]
pub struct Barrier<L: BarrierLevel, C: CubePrimitive> {
    _level: PhantomData<L>,
    _c: PhantomData<C>,
}

impl<L: BarrierLevel, C: CubePrimitive> IntoRuntime for Barrier<L, C> {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
        panic!("Doesn't exist at runtime")
    }
}

impl<L: BarrierLevel, C: CubePrimitive> CubeType for Barrier<L, C> {
    type ExpandType = BarrierExpand<L, C>;
}

impl<L, C: CubePrimitive> Init for BarrierExpand<L, C> {
    fn init(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<L, C: CubePrimitive> CubeDebug for BarrierExpand<L, C> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.elem, name);
    }
}

#[derive(Clone)]
/// Expand type of [Barrier]
pub struct BarrierExpand<L, C: CubePrimitive> {
    elem: ExpandElement,
    _level: PhantomData<L>,
    _c: PhantomData<C>,
}

impl<L: BarrierLevel, C: CubePrimitive> Default for Barrier<L, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: BarrierLevel, C: CubePrimitive> Barrier<L, C> {
    /// Create a barrier instance
    pub fn new() -> Self {
        Self {
            _level: PhantomData,
            _c: PhantomData,
        }
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async(&self, _source: Slice<Line<C>>, _destination: SliceMut<Line<C>>) {
        unexpanded!()
    }

    /// Wait until all data is loaded
    pub fn wait(&self) {
        unexpanded!()
    }

    pub fn __expand_new(scope: &mut Scope) -> BarrierExpand<L, C> {
        let elem = C::as_elem(scope);
        let variable = scope.create_barrier(Item::new(elem));
        BarrierExpand {
            elem: variable,
            _level: PhantomData,
            _c: PhantomData,
        }
    }

    pub fn __expand_memcpy_async(
        scope: &mut Scope,
        expand: BarrierExpand<L, C>,
        source: ExpandElementTyped<Slice<Line<C>>>,
        destination: ExpandElementTyped<SliceMut<Line<C>>>,
    ) {
        expand.__expand_memcpy_async_method(scope, source, destination);
    }

    pub fn __expand_wait(scope: &mut Scope, expand: BarrierExpand<L, C>) {
        expand.__expand_wait_method(scope);
    }
}

impl<L: BarrierLevel, C: CubePrimitive> BarrierExpand<L, C> {
    pub fn __expand_memcpy_async_method(
        &self,
        scope: &mut Scope,
        source: ExpandElementTyped<Slice<Line<C>>>,
        destination: ExpandElementTyped<SliceMut<Line<C>>>,
    ) {
        let barrier = *self.elem;
        let source = *source.expand;
        let destination = *destination.expand;

        let mem_copy = BarrierOps::MemCopyAsync {
            barrier,
            source,
            destination,
        };

        scope.register(mem_copy);
    }

    pub fn __expand_wait_method(&self, scope: &mut Scope) {
        let barrier = *self.elem;
        scope.register(BarrierOps::Wait { barrier });
    }
}
