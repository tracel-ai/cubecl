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

/// A mechanism for awaiting on asynchronous data transfers
/// Works at the Cube level
/// Or at the Unit level, using unit_count=1
#[derive(Clone, Copy)]
pub struct Barrier<C: CubePrimitive> {
    _c: PhantomData<C>,
}

impl<C: CubePrimitive> IntoRuntime for Barrier<C> {
    fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
        panic!("Doesn't exist at runtime")
    }
}

impl<C: CubePrimitive> CubeType for Barrier<C> {
    type ExpandType = BarrierExpand<C>;
}

impl<C: CubePrimitive> Init for BarrierExpand<C> {
    fn init(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<C: CubePrimitive> CubeDebug for BarrierExpand<C> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.elem, name);
    }
}

#[derive(Clone)]
/// Expand type of [Barrier]
pub struct BarrierExpand<C: CubePrimitive> {
    elem: ExpandElement,
    _c: PhantomData<C>,
}

impl<C: CubePrimitive> Barrier<C> {
    /// Create a barrier instance at the unit level, i.e. for the unit itself only
    pub fn new_unit_level() -> Self {
        Self { _c: PhantomData }
    }

    /// Create a barrier instance at the Cube level, i.e. for all units
    pub fn new_cube_level(_unit_count: u32) -> Self {
        Self { _c: PhantomData }
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

    pub fn __expand_new_unit_level(scope: &mut Scope) -> BarrierExpand<C> {
        Self::__expand_new_cube_level(scope, 1)
    }

    pub fn __expand_new_cube_level(scope: &mut Scope, unit_count: u32) -> BarrierExpand<C> {
        let elem = C::as_elem(scope);
        
        // For now we assume the elected unit is always the first one.
        let elected_unit = 0;
        let variable = scope.create_barrier(Item::new(elem), unit_count, elected_unit);
        BarrierExpand {
            elem: variable,
            _c: PhantomData,
        }
    }

    pub fn __expand_memcpy_async(
        scope: &mut Scope,
        expand: BarrierExpand<C>,
        source: ExpandElementTyped<Slice<Line<C>>>,
        destination: ExpandElementTyped<SliceMut<Line<C>>>,
    ) {
        expand.__expand_memcpy_async_method(scope, source, destination);
    }

    pub fn __expand_wait(scope: &mut Scope, expand: BarrierExpand<C>) {
        expand.__expand_wait_method(scope);
    }
}

impl<C: CubePrimitive> BarrierExpand<C> {
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
