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
/// Behaviour is defined by its [BarrierLevel](BarrierLevel).
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

pub struct BarrierLevel(InnerBarrierLevel);

#[derive(Clone)]
/// Defines how many units must reach the barrier to allow continuation
enum InnerBarrierLevel {
    /// Only waits for the unit who declared this barrier.
    /// This may be useful for waiting upon async data loading
    Unit,
    /// All units in the Cube must reach the barrier before continuing
    Cube(u32),
    /// Will use cooperative groups copy
    Cooperative,
}

impl BarrierLevel {
    /// Creates a Unit barrier level
    pub fn unit() -> Self {
        BarrierLevel(InnerBarrierLevel::Unit)
    }

    /// Creates a Cube barrier level
    ///
    /// The field elected_unit is the UNIT_POS of the unit that will
    /// perform the underlying initialization. Typically, 0 should work
    pub fn cube(elected_unit: u32) -> Self {
        BarrierLevel(InnerBarrierLevel::Cube(elected_unit))
    }

    /// Creates a Cooperative barrier level
    pub fn cooperative() -> Self {
        BarrierLevel(InnerBarrierLevel::Cooperative)
    }

    pub fn __expand_unit(_scope: &mut Scope) -> BarrierLevel {
        BarrierLevel(InnerBarrierLevel::Unit)
    }

    pub fn __expand_cube(_scope: &mut Scope, elected_unit: u32) -> Self {
        BarrierLevel(InnerBarrierLevel::Cube(elected_unit))
    }

    pub fn __expand_cooperative(_scope: &mut Scope) -> Self {
        BarrierLevel(InnerBarrierLevel::Cooperative)
    }
}

impl From<InnerBarrierLevel> for cubecl_ir::BarrierLevel {
    fn from(val: InnerBarrierLevel) -> Self {
        match val {
            InnerBarrierLevel::Unit => cubecl_ir::BarrierLevel::Unit,
            InnerBarrierLevel::Cube(elected_unit) => cubecl_ir::BarrierLevel::Cube(elected_unit),
            InnerBarrierLevel::Cooperative => cubecl_ir::BarrierLevel::Cooperative,
        }
    }
}

impl<C: CubePrimitive> Barrier<C> {
    /// Creates a barrier using a user defined comptime barrier level
    pub fn new(_level: BarrierLevel) -> Self {
        Self { _c: PhantomData }
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async(&self, _source: &Slice<Line<C>>, _destination: &mut SliceMut<Line<C>>) {
        unexpanded!()
    }

    /// Wait until all data is loaded
    pub fn wait(&self) {
        unexpanded!()
    }

    pub fn __expand_new(scope: &mut Scope, level: BarrierLevel) -> BarrierExpand<C> {
        let elem = C::as_elem(scope);

        let variable = scope.create_barrier(Item::new(elem), level.0.into());
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
