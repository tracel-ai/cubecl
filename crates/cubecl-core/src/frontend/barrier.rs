//! This module exposes barrier for asynchronous data transfer

use cubecl_ir::{ExpandElement, Instruction, Variable, VariableKind};
use paste::paste;

use crate::{
    ir::{BarrierOps, Scope},
    prelude::{CUBE_DIM, ExpandElementIntoMut},
    unexpanded,
};

use super::{
    CubeDebug, CubePrimitive, CubeType, ExpandElementTyped, IntoMut, Line, ReadOnly, ReadWrite,
    Slice, SliceExpand, SliceMut, TensorMap,
};

/// A mechanism for awaiting on asynchronous data transfers
/// Behaviour is defined by its [BarrierLevel](BarrierLevel).
#[derive(Clone, Copy)]
pub struct Barrier;

#[derive(Clone, Copy, PartialEq)]
pub struct BarrierToken;

impl CubeType for Barrier {
    type ExpandType = BarrierExpand;
}

impl IntoMut for BarrierExpand {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for BarrierExpand {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.elem, name);
    }
}

impl CubeType for BarrierToken {
    type ExpandType = ExpandElementTyped<BarrierToken>;
}

impl ExpandElementIntoMut for BarrierToken {
    fn elem_into_mut(_scope: &mut crate::ir::Scope, elem: ExpandElement) -> ExpandElement {
        elem
    }
}

#[derive(Clone)]
/// Expand type of [Barrier]
pub struct BarrierExpand {
    elem: ExpandElement,
}

#[derive(Clone)]
pub struct BarrierLevel(InnerBarrierLevel);

impl CubeType for BarrierLevel {
    type ExpandType = Self;
}

impl IntoMut for BarrierLevel {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl CubeDebug for BarrierLevel {
    fn set_debug_name(&self, _scope: &mut Scope, _name: &'static str) {}
}

#[derive(Clone)]
/// Defines how many units must reach the barrier before execution can continue.
/// This also determines how `memcpy_async` operations should be handled.
enum InnerBarrierLevel {
    /// Waits only for the unit that declared this barrier.
    /// Useful for synchronizing after async data loading.
    Unit,

    /// Only the leader unit is required to reach the barrier before continuing.
    /// The argument is the ID of the unit elected for initialization.
    ///
    /// TMA loads are issued from only a single unit, and this leader is the one that should arrive
    /// on the barrier. Unlike `Unit`, this barrier is *shared*, so all threads can wait on it.
    CubeUnit(ExpandElement),

    /// All units in the Cube must reach the barrier before continuing.
    /// The argument is the ID of the unit elected for initialization.
    CubeFull(ExpandElement),

    /// `arrival_count` units are required before the barrier can continue.
    /// The arguments are the ID of the unit elected for initialization, and the number of units
    /// that should call `arrive`.
    ///
    /// TMA loads are issued from only a single unit, and this leader is the one that should arrive
    /// on the barrier. Unlike `Unit`, this barrier is *shared*, so all threads can wait on it.
    CubeCustom {
        is_elected: ExpandElement,
        arrival_count: ExpandElement,
    },

    /// Fully manual Cube barrier, no automatic initialization
    CubeManual,
}

impl BarrierLevel {
    /// Creates a Unit barrier level
    pub fn unit() -> Self {
        BarrierLevel(InnerBarrierLevel::Unit)
    }

    /// Creates a CubeUnit barrier level
    ///
    /// Same as `cube_full` but with an expected arrival count of `1`. Only the leader thread will
    /// arrive on the barrier. Useful for TMA
    pub fn cube_unit(_is_elected: bool) -> Self {
        unexpanded!()
    }

    /// Creates a CubeCoop barrier level
    ///
    /// Will sync all units
    pub fn cube_full(_is_elected: bool) -> Self {
        unexpanded!()
    }

    /// Creates a CubeCustom barrier level
    ///
    /// Will sync `arrival_count` units
    pub fn cube_custom(_arrival_count: u32) -> Self {
        unexpanded!()
    }

    /// Creates a CubeManual barrier level
    /// Not initialized automatically
    pub fn cube_manual() -> Self {
        unexpanded!()
    }

    fn arrival_count(&self, scope: &mut Scope) -> Variable {
        match &self.0 {
            InnerBarrierLevel::Unit | InnerBarrierLevel::CubeUnit(_) => 1.into(),
            InnerBarrierLevel::CubeFull(_) => *CUBE_DIM::expand(scope).expand,
            InnerBarrierLevel::CubeCustom { arrival_count, .. } => **arrival_count,
            InnerBarrierLevel::CubeManual => panic!("Can't get arrival count of manual barrier"),
        }
    }

    fn is_elected(&self) -> Variable {
        match &self.0 {
            InnerBarrierLevel::Unit => true.into(),
            InnerBarrierLevel::CubeUnit(is_elected)
            | InnerBarrierLevel::CubeFull(is_elected)
            | InnerBarrierLevel::CubeCustom { is_elected, .. } => **is_elected,
            InnerBarrierLevel::CubeManual => panic!("Can't get `is_elected` of manual barrier"),
        }
    }

    pub fn __expand_unit(_scope: &mut Scope) -> BarrierLevel {
        BarrierLevel(InnerBarrierLevel::Unit)
    }

    pub fn __expand_cube_unit(_scope: &mut Scope, is_elected: ExpandElementTyped<bool>) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeUnit(is_elected.expand))
    }

    pub fn __expand_cube_full(_scope: &mut Scope, is_elected: ExpandElementTyped<bool>) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeFull(is_elected.expand))
    }

    pub fn __expand_cube_custom(
        _scope: &mut Scope,
        is_elected: ExpandElementTyped<bool>,
        arrival_count: ExpandElementTyped<u32>,
    ) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeCustom {
            is_elected: is_elected.expand,
            arrival_count: arrival_count.expand,
        })
    }

    pub fn __expand_cube_manual(_scope: &mut Scope) -> Self {
        BarrierLevel(InnerBarrierLevel::CubeManual)
    }
}

impl From<InnerBarrierLevel> for cubecl_ir::BarrierLevel {
    fn from(val: InnerBarrierLevel) -> Self {
        match val {
            InnerBarrierLevel::Unit => cubecl_ir::BarrierLevel::Unit,
            InnerBarrierLevel::CubeUnit(_)
            | InnerBarrierLevel::CubeFull(_)
            | InnerBarrierLevel::CubeCustom { .. }
            | InnerBarrierLevel::CubeManual => cubecl_ir::BarrierLevel::Cube,
        }
    }
}

macro_rules! tensor_map_load {
    ($dim: literal, $($arg: expr),*) => {
        paste! {
            impl Barrier {
                /// Copy a tile from a global memory `source` to a shared memory `destination`, with
                /// the provided offsets.
                #[allow(unused, clippy::too_many_arguments)]
                pub fn [<tma_load_ $dim d>]<C: CubePrimitive>(
                    &self,
                    source: &TensorMap<C>,
                    destination: &mut SliceMut<Line<C>>,
                    $($arg: i32),*
                ) {
                    unexpanded!()
                }

                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_ $dim d>]<C: CubePrimitive>(
                    scope: &mut Scope,
                    expand: BarrierExpand,
                    source: ExpandElementTyped<TensorMap<C>>,
                    destination: SliceExpand<Line<C>, ReadWrite>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    expand.[<__expand_tma_load_ $dim d_method>](scope, source, destination, $($arg),*);
                }
            }

            impl BarrierExpand {
                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_ $dim d_method>]<C: CubePrimitive>(
                    &self,
                    scope: &mut Scope,
                    source: ExpandElementTyped<TensorMap<C>>,
                    destination: SliceExpand<Line<C>, ReadWrite>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    let barrier = *self.elem;
                    let source = *source.expand;
                    let (destination, destination_offset) = destination.__to_raw_parts();

                    let mem_copy = BarrierOps::TmaLoad {
                        barrier,
                        tensor_map: source,
                        indices: vec![$(*$arg.expand),*],
                        offset_out: destination_offset
                    };

                    scope.register(Instruction::new(mem_copy, destination));
                }
            }
        }
    };
}

macro_rules! tensor_map_load_im2col {
    ($dim: literal, $($arg: expr),*; $($offset: expr),*) => {
        paste! {
            impl Barrier {
                /// Copy a tile from a global memory `source` to a shared memory `destination`, with
                /// the provided offsets.
                #[allow(unused, clippy::too_many_arguments)]
                pub fn [<tma_load_im2col_ $dim d>]<C: CubePrimitive>(
                    &self,
                    source: &TensorMap<C>,
                    destination: &mut SliceMut<Line<C>>,
                    $($arg: i32,)*
                    $($offset: u16),*
                ) {
                    unexpanded!()
                }

                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_im2col_ $dim d>]<C: CubePrimitive>(
                    scope: &mut Scope,
                    expand: BarrierExpand,
                    source: ExpandElementTyped<TensorMap<C>>,
                    destination: SliceExpand<Line<C>, ReadWrite>,
                    $($arg: ExpandElementTyped<i32>,)*
                    $($offset: ExpandElementTyped<u16>),*
                ) {
                    expand.[<__expand_tma_load_im2col_ $dim d_method>](scope, source, destination, $($arg),*, $($offset),*);
                }
            }

            impl BarrierExpand {
                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_im2col_ $dim d_method>]<C: CubePrimitive>(
                    &self,
                    scope: &mut Scope,
                    source: ExpandElementTyped<TensorMap<C>>,
                    destination: SliceExpand<Line<C>, ReadWrite>,
                    $($arg: ExpandElementTyped<i32>,)*
                    $($offset: ExpandElementTyped<u16>),*
                ) {
                    let barrier = *self.elem;
                    let source = *source.expand;
                    let (destination, destination_offset) = destination.__to_raw_parts();

                    let mem_copy = BarrierOps::TmaLoadIm2col {
                        barrier,
                        tensor_map: source,
                        indices: vec![$(*$arg.expand),*],
                        offsets: vec![$(*$offset.expand),*],
                        offset_out: destination_offset,
                    };

                    scope.register(Instruction::new(mem_copy, destination));
                }
            }
        }
    };
}

tensor_map_load!(1, x);
tensor_map_load!(2, y, x);
tensor_map_load!(3, z, y, x);
tensor_map_load!(4, w, z, y, x);
tensor_map_load!(5, v, w, z, y, x);

tensor_map_load_im2col!(3, n, w, c; w_offset);
tensor_map_load_im2col!(4, n, h, w, c; h_offset, w_offset);
tensor_map_load_im2col!(5, n, d, h, w, c; d_offset, h_offset, w_offset);

impl Barrier {
    /// Creates a barrier using a user defined comptime barrier level
    pub fn new(_level: BarrierLevel) -> Self {
        Self
    }

    /// Creates a new barrier for use with TMA instructions. Adds a shared memory proxy barrier to
    /// the initialization.
    pub fn new_with_async_proxy_fence(_level: BarrierLevel) -> Self {
        Self
    }

    /// Manually initialize the barrier, without handling synchronization, etc.
    pub fn init_manual(&self, _arrival_count: u32) -> BarrierToken {
        unexpanded!()
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async<C: CubePrimitive>(
        &self,
        _source: &Slice<Line<C>>,
        _destination: &mut SliceMut<Line<C>>,
    ) {
        unexpanded!()
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async_cooperative<C: CubePrimitive>(
        &self,
        _source: &Slice<Line<C>>,
        _destination: &mut SliceMut<Line<C>>,
    ) {
        unexpanded!()
    }

    /// Copy the source slice to destination. Uses transaction count like TMA, so use with
    /// `expect_tx` or `arrive_and_expect_tx`.
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async_tx<C: CubePrimitive>(
        &self,
        _source: &Slice<Line<C>>,
        _destination: &mut SliceMut<Line<C>>,
    ) {
        unexpanded!()
    }

    /// Arrive at the barrier, decrementing arrival count
    pub fn arrive(&self) -> BarrierToken {
        unexpanded!()
    }

    /// Makes all previous `copy_async` operations visible on the barrier.
    /// Does *not* count as an arrive in terms of the barrier arrival count. So `arrive` or
    /// `arrive_and_wait` should still be called afterwards.
    pub fn commit_copy_async(&self) {
        unexpanded!()
    }

    /// Arrive at the barrier, decrementing arrival count. Additionally increments expected count.
    pub fn arrive_and_expect_tx(
        &self,
        _arrival_count: u32,
        _transaction_count: u32,
    ) -> BarrierToken {
        unexpanded!()
    }

    /// Increments the expected count of the barrier.
    pub fn expect_tx(&self, _expected_count: u32) {
        unexpanded!()
    }

    /// Wait at the barrier until all arrivals are done
    pub fn wait(&self, _token: BarrierToken) {
        unexpanded!()
    }

    /// Wait at the barrier until the `phase` is completed. Doesn't require a token, but needs phase
    /// to be managed manually.
    pub fn wait_parity(&self, _phase: u32) {
        unexpanded!()
    }

    /// Wait until all data is loaded
    pub fn arrive_and_wait(&self) {
        unexpanded!()
    }

    pub fn __expand_new(scope: &mut Scope, level: BarrierLevel) -> BarrierExpand {
        let variable = scope.create_barrier(level.0.clone().into());
        match &level.0 {
            InnerBarrierLevel::CubeManual => {
                scope.register(BarrierOps::Declare { barrier: *variable });
            }
            _ => {
                let is_elected = level.is_elected();
                let arrival_count = level.arrival_count(scope);
                scope.register(BarrierOps::Init {
                    barrier: *variable,
                    is_elected,
                    arrival_count,
                    with_async_proxy_fence: false,
                });
            }
        }

        BarrierExpand { elem: variable }
    }

    pub fn __expand_new_with_async_proxy_fence(
        scope: &mut Scope,
        level: BarrierLevel,
    ) -> BarrierExpand {
        let is_elected = level.is_elected();
        let arrival_count = level.arrival_count(scope);
        let variable = scope.create_barrier(level.0.clone().into());
        scope.register(BarrierOps::Init {
            barrier: *variable,
            is_elected,
            arrival_count,
            with_async_proxy_fence: true,
        });
        BarrierExpand { elem: variable }
    }

    pub fn __expand_init_manual(
        scope: &mut Scope,
        expand: BarrierExpand,
        arrival_count: ExpandElementTyped<u32>,
    ) {
        expand.__expand_init_manual_method(scope, arrival_count);
    }

    pub fn __expand_memcpy_async<C: CubePrimitive>(
        scope: &mut Scope,
        expand: BarrierExpand,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
    ) {
        expand.__expand_memcpy_async_method(scope, source, destination);
    }

    pub fn __expand_memcpy_async_cooperative<C: CubePrimitive>(
        scope: &mut Scope,
        expand: BarrierExpand,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
    ) {
        expand.__expand_memcpy_async_method(scope, source, destination);
    }

    pub fn __expand_memcpy_async_tx<C: CubePrimitive>(
        scope: &mut Scope,
        expand: BarrierExpand,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
    ) {
        expand.__expand_memcpy_async_tx_method(scope, source, destination);
    }

    pub fn __expand_arrive(
        scope: &mut Scope,
        expand: BarrierExpand,
    ) -> ExpandElementTyped<BarrierToken> {
        expand.__expand_arrive_method(scope)
    }

    pub fn __expand_commit_copy_async(scope: &mut Scope, expand: BarrierExpand) {
        expand.__expand_commit_copy_async_method(scope)
    }

    pub fn __expand_arrive_and_expect_tx(
        scope: &mut Scope,
        expand: BarrierExpand,
        arrival_count: ExpandElementTyped<u32>,
        transaction_count: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<BarrierToken> {
        expand.__expand_arrive_and_expect_tx_method(scope, arrival_count, transaction_count)
    }

    pub fn __expand_expect_tx(
        scope: &mut Scope,
        expand: BarrierExpand,
        expected_count: ExpandElementTyped<u32>,
    ) {
        expand.__expand_expect_tx_method(scope, expected_count);
    }

    pub fn __expand_wait(
        scope: &mut Scope,
        expand: BarrierExpand,
        token: ExpandElementTyped<BarrierToken>,
    ) {
        expand.__expand_wait_method(scope, token);
    }

    pub fn __expand_wait_parity(
        scope: &mut Scope,
        expand: BarrierExpand,
        phase: ExpandElementTyped<u32>,
    ) {
        expand.__expand_wait_parity_method(scope, phase);
    }

    pub fn __expand_arrive_and_wait(scope: &mut Scope, expand: BarrierExpand) {
        expand.__expand_arrive_and_wait_method(scope);
    }
}

impl BarrierExpand {
    pub fn __expand_init_manual_method(
        &self,
        scope: &mut Scope,
        arrival_count: ExpandElementTyped<u32>,
    ) {
        let barrier = *self.elem;

        scope.register(BarrierOps::InitManual {
            barrier,
            arrival_count: *arrival_count.expand,
        });
    }

    pub fn __expand_memcpy_async_method<C: CubePrimitive>(
        &self,
        scope: &mut Scope,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
    ) {
        let barrier = *self.elem;
        let source_length = *source.length.expand;
        let (source, source_offset) = source.__to_raw_parts();
        let (destination, destination_offset) = destination.__to_raw_parts();

        let mem_copy = BarrierOps::MemCopyAsync {
            barrier,
            source,
            source_length,
            offset_source: source_offset,
            offset_out: destination_offset,
        };

        scope.register(Instruction::new(mem_copy, destination));
    }

    pub fn __expand_memcpy_async_cooperative_method<C: CubePrimitive>(
        &self,
        scope: &mut Scope,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
    ) {
        let barrier = *self.elem;
        let source_length = *source.length.expand;
        let (source, source_offset) = source.__to_raw_parts();
        let (destination, destination_offset) = destination.__to_raw_parts();

        let mem_copy = BarrierOps::MemCopyAsyncCooperative {
            barrier,
            source,
            source_length,
            offset_source: source_offset,
            offset_out: destination_offset,
        };

        scope.register(Instruction::new(mem_copy, destination));
    }

    pub fn __expand_memcpy_async_tx_method<C: CubePrimitive>(
        &self,
        scope: &mut Scope,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
    ) {
        let barrier = *self.elem;
        let source_length = *source.length.expand;
        let (source, source_offset) = source.__to_raw_parts();
        let (destination, destination_offset) = destination.__to_raw_parts();

        let mem_copy = BarrierOps::MemCopyAsyncTx {
            barrier,
            source,
            source_length,
            offset_source: source_offset,
            offset_out: destination_offset,
        };

        scope.register(Instruction::new(mem_copy, destination));
    }

    pub fn __expand_arrive_method(&self, scope: &mut Scope) -> ExpandElementTyped<BarrierToken> {
        let barrier = *self.elem;
        let VariableKind::Barrier { id, level, .. } = barrier.kind else {
            unreachable!()
        };
        let token = scope.create_barrier_token(id, level);
        scope.register(Instruction::new(BarrierOps::Arrive { barrier }, *token));
        token.into()
    }

    pub fn __expand_commit_copy_async_method(&self, scope: &mut Scope) {
        let barrier = *self.elem;
        let VariableKind::Barrier { id, level, .. } = barrier.kind else {
            unreachable!()
        };
        let token = scope.create_barrier_token(id, level);
        scope.register(Instruction::new(
            BarrierOps::CommitCopyAsync { barrier },
            *token,
        ));
    }

    pub fn __expand_arrive_and_expect_tx_method(
        &self,
        scope: &mut Scope,
        arrival_count: ExpandElementTyped<u32>,
        transaction_count: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<BarrierToken> {
        let barrier = *self.elem;
        let VariableKind::Barrier { id, level, .. } = barrier.kind else {
            unreachable!()
        };
        let token = scope.create_barrier_token(id, level);
        let arrival_count: ExpandElement = arrival_count.into();
        let transaction_count: ExpandElement = transaction_count.into();
        scope.register(Instruction::new(
            BarrierOps::ArriveTx {
                barrier,
                arrive_count_update: arrival_count.consume(),
                transaction_count_update: transaction_count.consume(),
            },
            *token,
        ));
        token.into()
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

    pub fn __expand_wait_method(&self, scope: &mut Scope, token: ExpandElementTyped<BarrierToken>) {
        let barrier = *self.elem;
        let token = *token.expand;
        scope.register(BarrierOps::Wait { barrier, token });
    }

    pub fn __expand_wait_parity_method(&self, scope: &mut Scope, phase: ExpandElementTyped<u32>) {
        let barrier = *self.elem;
        let phase = *phase.expand;
        scope.register(BarrierOps::WaitParity { barrier, phase });
    }

    pub fn __expand_arrive_and_wait_method(&self, scope: &mut Scope) {
        let barrier = *self.elem;
        scope.register(BarrierOps::ArriveAndWait { barrier });
    }
}

/// Copy the source slice in global memory to destination in shared memory with a low level async
/// copy. This only copies up to 128 bits/16 bytes, and does not synchronize. Use
/// `barrier.copy_async_arrive` to make the reads visible.
/// `copy_size` is in terms of elements to simplify copying between different line sizes.
///
/// # Safety
///
/// This will try to copy the whole source slice, so
/// make sure source length <= destination length
pub fn copy_async<C: CubePrimitive>(
    _source: &Slice<Line<C>>,
    _destination: &mut SliceMut<Line<C>>,
    _copy_size: u32,
) {
    unexpanded!()
}

pub mod copy_async {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
        copy_length: u32,
    ) {
        let source_length = copy_length.into();
        let (source, source_offset) = source.__to_raw_parts();
        let (destination, destination_offset) = destination.__to_raw_parts();

        let mem_copy = BarrierOps::CopyAsync {
            source,
            source_length,
            offset_source: source_offset,
            offset_out: destination_offset,
            copy_length: copy_length * C::as_type(scope).size() as u32,
            checked: false,
        };

        scope.register(Instruction::new(mem_copy, destination));
    }
}

/// Copy the source slice in global memory to destination in shared memory with a low level async
/// copy. This only copies up to 128 bits/16 bytes, and does not synchronize. Use
/// `barrier.copy_async_arrive` to make the reads visible.
/// `copy_size` is in terms of elements to simplify copying between different line sizes.
///
/// Will only copy the length of the source slice, and zero fill the rest.
pub fn copy_async_checked<C: CubePrimitive>(
    _source: &Slice<Line<C>>,
    _destination: &mut SliceMut<Line<C>>,
    _copy_size: u32,
) {
    unexpanded!();
}

pub mod copy_async_checked {
    use super::*;

    pub fn expand<C: CubePrimitive>(
        scope: &mut Scope,
        source: SliceExpand<Line<C>, ReadOnly>,
        destination: SliceExpand<Line<C>, ReadWrite>,
        copy_length: u32,
    ) {
        let source_length = *source.length.expand;
        let (source, source_offset) = source.__to_raw_parts();
        let (destination, destination_offset) = destination.__to_raw_parts();

        let mem_copy = BarrierOps::CopyAsync {
            source,
            source_length,
            offset_source: source_offset,
            offset_out: destination_offset,
            copy_length: copy_length * C::as_type(scope).size() as u32,
            checked: true,
        };

        scope.register(Instruction::new(mem_copy, destination));
    }
}
