//! This module exposes barrier for asynchronous data transfer

use std::marker::PhantomData;

use cubecl_ir::{ExpandElement, Instruction};
use paste::paste;

use crate::{
    ir::{BarrierOps, Item, Scope},
    unexpanded,
};

use super::{
    CubeDebug, CubePrimitive, CubeType, ExpandElementTyped, Init, Line, Slice, SliceMut, TensorMap,
};

/// A mechanism for awaiting on asynchronous data transfers
/// Behaviour is defined by its [BarrierLevel](BarrierLevel).
#[derive(Clone, Copy)]
pub struct Barrier<C: CubePrimitive> {
    _c: PhantomData<C>,
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

macro_rules! tensor_map_copy_to_shared {
    ($dim: literal, $($arg: expr),*) => {
        paste! {
            impl<C: CubePrimitive> Barrier<C> {
                /// Copy a tile from a global memory `source` to a shared memory `destination`, with
                /// the provided offsets.
                #[allow(unused, clippy::too_many_arguments)]
                pub fn [<memcpy_async_tensor_to_shared_ $dim d>](
                    &self,
                    source: &TensorMap<C>,
                    destination: &mut SliceMut<Line<C>>,
                    $($arg: i32),*
                ) {
                    unexpanded!()
                }

                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_memcpy_async_tensor_to_shared_ $dim d>](
                    scope: &mut Scope,
                    expand: BarrierExpand<C>,
                    source: ExpandElementTyped<TensorMap<C>>,
                    destination: ExpandElementTyped<SliceMut<Line<C>>>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    expand.[<__expand_memcpy_async_tensor_to_shared_ $dim d_method>](scope, source, destination, $($arg),*);
                }
            }

            impl<C: CubePrimitive> BarrierExpand<C> {
                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_memcpy_async_tensor_to_shared_ $dim d_method>](
                    &self,
                    scope: &mut Scope,
                    source: ExpandElementTyped<TensorMap<C>>,
                    destination: ExpandElementTyped<SliceMut<Line<C>>>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    let barrier = *self.elem;
                    let source = *source.expand;
                    let destination = *destination.expand;

                    let mem_copy = BarrierOps::MemCopyAsyncTensorGlobalToShared {
                        barrier,
                        tensor_map: source,
                        indices: vec![$(*$arg.expand),*],
                    };

                    scope.register(Instruction::new(mem_copy, destination));
                }
            }
        }
    };
}

tensor_map_copy_to_shared!(2, y, x);
tensor_map_copy_to_shared!(3, z, y, x);
tensor_map_copy_to_shared!(4, w, z, y, x);
tensor_map_copy_to_shared!(5, v, w, z, y, x);

impl<C: CubePrimitive> Barrier<C> {
    /// Creates a barrier using a user defined comptime barrier level
    pub fn new(_level: BarrierLevel) -> Self {
        Self { _c: PhantomData }
    }

    /// Creates a new barrier for use with TMA instructions. Adds a shared memory proxy barrier to
    /// the initialization.
    pub fn new_with_tma_proxy(_level: BarrierLevel) -> Self {
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

    /// Arrive at the barrier, decrementing arrival count
    pub fn arrive(&self) {
        unexpanded!()
    }

    /// Arrive at the barrier, decrementing arrival count. Additionally increments expected count.
    pub fn arrive_tx(&self, _arrival_count: u32, _transaction_count: u32) {
        unexpanded!()
    }

    /// Increments the expected count of the barrier.
    pub fn expect_tx(&self, _expected_count: u32) {
        unexpanded!()
    }

    /// Wait at the barrier until all arrivals are done
    pub fn wait(&self) {
        unexpanded!()
    }

    /// Wait until all data is loaded
    pub fn arrive_and_wait(&self) {
        unexpanded!()
    }

    pub fn __expand_new(scope: &mut Scope, level: BarrierLevel) -> BarrierExpand<C> {
        let elem = C::as_elem(scope);

        let variable = scope.create_barrier(Item::new(elem), level.0.into());
        scope.register(BarrierOps::Init {
            barrier: *variable,
            with_cta_fence: false,
        });
        BarrierExpand {
            elem: variable,
            _c: PhantomData,
        }
    }

    pub fn __expand_new_with_tma_proxy(scope: &mut Scope, level: BarrierLevel) -> BarrierExpand<C> {
        let elem = C::as_elem(scope);

        let variable = scope.create_barrier(Item::new(elem), level.0.into());
        scope.register(BarrierOps::Init {
            barrier: *variable,
            with_cta_fence: true,
        });
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

    pub fn __expand_arrive(scope: &mut Scope, expand: BarrierExpand<C>) {
        expand.__expand_arrive_method(scope);
    }

    pub fn __expand_arrive_tx(
        scope: &mut Scope,
        expand: BarrierExpand<C>,
        arrival_count: ExpandElementTyped<u32>,
        transaction_count: ExpandElementTyped<u32>,
    ) {
        expand.__expand_arrive_tx_method(scope, arrival_count, transaction_count);
    }

    pub fn __expand_expect_tx(
        scope: &mut Scope,
        expand: BarrierExpand<C>,
        expected_count: ExpandElementTyped<u32>,
    ) {
        expand.__expand_expect_tx_method(scope, expected_count);
    }

    pub fn __expand_wait(scope: &mut Scope, expand: BarrierExpand<C>) {
        expand.__expand_wait_method(scope);
    }

    pub fn __expand_arrive_and_wait(scope: &mut Scope, expand: BarrierExpand<C>) {
        expand.__expand_arrive_and_wait_method(scope);
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

        let mem_copy = BarrierOps::MemCopyAsync { barrier, source };

        scope.register(Instruction::new(mem_copy, destination));
    }

    pub fn __expand_arrive_method(&self, scope: &mut Scope) {
        let barrier = *self.elem;
        scope.register(BarrierOps::Arrive { barrier });
    }

    pub fn __expand_arrive_tx_method(
        &self,
        scope: &mut Scope,
        arrival_count: ExpandElementTyped<u32>,
        transaction_count: ExpandElementTyped<u32>,
    ) {
        let barrier = *self.elem;
        let arrival_count: ExpandElement = arrival_count.into();
        let transaction_count: ExpandElement = transaction_count.into();
        scope.register(BarrierOps::ArriveTx {
            barrier,
            arrive_count_update: arrival_count.consume(),
            transaction_count_update: transaction_count.consume(),
        });
    }

    pub fn __expand_expect_tx_method(
        &self,
        scope: &mut Scope,
        transaction_count: ExpandElementTyped<u32>,
    ) {
        let barrier = *self.elem;
        let transaction_count: ExpandElement = transaction_count.into();
        scope.register(BarrierOps::ExpectTx {
            barrier,
            transaction_count_update: transaction_count.consume(),
        });
    }

    pub fn __expand_wait_method(&self, scope: &mut Scope) {
        let barrier = *self.elem;
        scope.register(BarrierOps::Wait { barrier });
    }

    pub fn __expand_arrive_and_wait_method(&self, scope: &mut Scope) {
        let barrier = *self.elem;
        scope.register(BarrierOps::ArriveAndWait { barrier });
    }
}
