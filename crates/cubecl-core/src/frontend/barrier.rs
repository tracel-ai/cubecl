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

#[derive(Clone, PartialEq, Eq)]
pub struct BarrierLevel(InnerBarrierLevel);

impl CubeType for BarrierLevel {
    type ExpandType = Self;
}

impl Init for BarrierLevel {
    fn init(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for BarrierLevel {
    fn set_debug_name(&self, _scope: &mut Scope, _name: &'static str) {}
}

#[derive(Clone, Eq, PartialEq)]
/// Defines how many units must reach the barrier before execution can continue.
/// This also determines how `memcpy_async` operations should be handled.
enum InnerBarrierLevel {
    /// Waits only for the unit that declared this barrier.
    /// Useful for synchronizing after async data loading.
    Unit,

    /// All units in the Cube must reach the barrier before continuing.
    /// The argument is the ID of the unit elected for initialization.
    ///
    /// `memcpy_async` is **cooperative**, so all units in the Cube must call `memcpy_async` with the same arguments.
    /// The called is not elected by default, so it must be done manually if wanted
    CubeCoop(u32),

    /// All units in the Cube must reach the barrier before continuing.
    /// The argument is the ID of the unit elected for initialization.
    ///
    /// `memcpy_async` is **not cooperative**, so each unit must manually handle its own data slice.
    CubeManual(u32),
}

impl BarrierLevel {
    /// Creates a Unit barrier level
    pub fn unit() -> Self {
        BarrierLevel(InnerBarrierLevel::Unit)
    }

    /// Creates a CubeCoop barrier level
    ///
    /// Will sync all units
    pub fn cube_coop(elected_unit: u32) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeCoop(elected_unit))
    }

    /// Creates a CubeManual barrier level
    ///
    /// Will sync all units
    pub fn cube_manual(elected_unit: u32) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeManual(elected_unit))
    }

    pub fn __expand_unit(_scope: &mut Scope) -> BarrierLevel {
        BarrierLevel(InnerBarrierLevel::Unit)
    }

    pub fn __expand_cube_coop(_scope: &mut Scope, elected_unit: u32) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeCoop(elected_unit))
    }

    pub fn __expand_cube_manual(_scope: &mut Scope, elected_unit: u32) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeManual(elected_unit))
    }
}

impl From<InnerBarrierLevel> for cubecl_ir::BarrierLevel {
    fn from(val: InnerBarrierLevel) -> Self {
        match val {
            InnerBarrierLevel::Unit => cubecl_ir::BarrierLevel::Unit,
            InnerBarrierLevel::CubeCoop(elected_unit) => {
                cubecl_ir::BarrierLevel::CubeCoop(elected_unit)
            }
            InnerBarrierLevel::CubeManual(elected_unit) => {
                cubecl_ir::BarrierLevel::CubeManual(elected_unit)
            }
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
