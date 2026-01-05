//! This module exposes barrier for asynchronous data transfer

use std::ops::{Deref, DerefMut};

use crate as cubecl;
use cubecl_ir::{ExpandElement, Instruction, OpaqueType};
use cubecl_macros::intrinsic;
use paste::paste;

use crate::{
    ir::{BarrierOps, Scope},
    prelude::*,
    unexpanded,
};

use super::{
    CubePrimitive, CubeType, ExpandElementTyped, Line, ReadOnly, ReadWrite, Slice, SliceExpand,
    SliceMut, TensorMap,
};

/// A mechanism for awaiting on asynchronous data transfers
/// Behaviour is defined by its [BarrierLevel](BarrierLevel).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Barrier;
pub type BarrierExpand = ExpandElementTyped<Barrier>;

#[derive(Clone, Copy, PartialEq)]
pub struct BarrierToken;

impl CubeType for Barrier {
    type ExpandType = ExpandElementTyped<Barrier>;
}

impl CubePrimitive for Barrier {
    fn from_const_value(_value: cubecl_ir::ConstantValue) -> Self {
        unreachable!("Can't create from const value")
    }
}

impl ExpandElementIntoMut for Barrier {
    fn elem_into_mut(_scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        elem
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

macro_rules! tensor_map_load {
    ($dim: literal, $($arg: expr),*) => {
        paste! {
            impl Barrier {
                /// Copy a tile from a global memory `source` to a shared memory `destination`, with
                /// the provided offsets.
                #[allow(unused, clippy::too_many_arguments)]
                pub fn [<tma_load_ $dim d>]<C: CubePrimitive>(
                    &self,
                    source: &TensorMap<C, Tiled>,
                    destination: &mut SliceMut<Line<C>>,
                    $($arg: i32),*
                ) {
                    unexpanded!()
                }

                #[allow(clippy::too_many_arguments)]
                pub fn [<__expand_tma_load_ $dim d>]<C: CubePrimitive>(
                    scope: &mut Scope,
                    expand: BarrierExpand,
                    source: ExpandElementTyped<TensorMap<C, Tiled>>,
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
                    source: ExpandElementTyped<TensorMap<C, Tiled>>,
                    destination: SliceExpand<Line<C>, ReadWrite>,
                    $($arg: ExpandElementTyped<i32>),*
                ) {
                    let barrier = *self.expand;
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
                    source: &TensorMap<C, Im2col>,
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
                    source: ExpandElementTyped<TensorMap<C, Im2col>>,
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
                    source: ExpandElementTyped<TensorMap<C, Im2col>>,
                    destination: SliceExpand<Line<C>, ReadWrite>,
                    $($arg: ExpandElementTyped<i32>,)*
                    $($offset: ExpandElementTyped<u16>),*
                ) {
                    let barrier = *self.expand;
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

#[cube(self_type = "ref")]
impl Barrier {
    /// Create a local barrier object for the current unit. Automatically initialized with an
    /// arrival count of `1`.
    pub fn local() -> Self {
        intrinsic!(|scope| {
            let variable =
                scope.create_local_mut(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Unit));
            scope.register(BarrierOps::Init {
                barrier: *variable,
                is_elected: true.into(),
                arrival_count: 1.into(),
            });
            variable.into()
        })
    }

    /// Create a shared memory barrier that can be accesses by all units in the cube. Initialized
    /// by the `is_elected` unit with an arrival count of `arrival_count`. This is the number of
    /// times `arrive` or one of its variants needs to be called before the barrier advances.
    ///
    /// If all units in the cube arrive on the barrier, use `CUBE_DIM` as the arrival count. For
    /// other purposes, only a subset may need to arrive.
    #[allow(unused_variables)]
    pub fn shared(arrival_count: u32, is_elected: bool) -> Shared<Barrier> {
        intrinsic!(|scope| {
            let variable = scope.create_shared(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Cube));
            scope.register(BarrierOps::Init {
                barrier: *variable,
                is_elected: *is_elected.expand,
                arrival_count: *arrival_count.expand,
            });
            variable.into()
        })
    }

    /// Create a shared memory barrier that can be accesses by all units in the cube. Only declared,
    /// but not initialized.
    pub fn shared_uninit() -> Shared<Barrier> {
        intrinsic!(|scope| {
            let variable = scope.create_shared(OpaqueType::Barrier(cubecl_ir::BarrierLevel::Cube));
            scope.register(BarrierOps::Declare { barrier: *variable });
            variable.into()
        })
    }

    /// Initializes a barrier with a given `arrival_count`. This is the number of
    /// times `arrive` or one of its variants needs to be called before the barrier advances.
    ///
    /// If all units in the cube arrive on the barrier, use `CUBE_DIM` as the arrival count. For
    /// other purposes, only a subset may need to arrive.
    ///
    /// # Note
    ///
    /// No synchronization or election is performed, this is raw initialization. For shared barriers
    /// ensure only one unit performs the initialization, and synchronize the cube afterwards. There
    /// may also be additional synchronization requirements for bulk copy operations, like
    /// [`sync_async_proxy_shared()`].
    #[allow(unused_variables)]
    pub fn init_manual(&self, arrival_count: u32) {
        intrinsic!(|scope| {
            let barrier = *self.expand.clone();

            scope.register(BarrierOps::InitManual {
                barrier,
                arrival_count: *arrival_count.expand,
            });
        })
    }
}

// MemcpyAsync

#[cube(self_type = "ref")]
impl Barrier {
    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    #[allow(unused_variables)]
    pub fn memcpy_async<C: CubePrimitive>(
        &self,
        source: &Slice<Line<C>>,
        destination: &mut SliceMut<Line<C>>,
    ) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
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
        })
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    #[allow(unused_variables)]
    pub fn memcpy_async_cooperative<C: CubePrimitive>(
        &self,
        source: &Slice<Line<C>>,
        destination: &mut SliceMut<Line<C>>,
    ) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
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
        })
    }

    /// Copy the source slice to destination. Uses transaction count like TMA, so use with
    /// `expect_tx` or `arrive_and_expect_tx`.
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    #[allow(unused_variables)]
    pub fn memcpy_async_tx<C: CubePrimitive>(
        &self,
        source: &Slice<Line<C>>,
        destination: &mut SliceMut<Line<C>>,
    ) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
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
        })
    }
}

// Arrival and Wait

#[cube(self_type = "ref")]
impl Barrier {
    /// Arrive at the barrier, decrementing arrival count
    pub fn arrive(&self) -> BarrierToken {
        intrinsic!(|scope| {
            let barrier = *self.expand;
            let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type() else {
                unreachable!()
            };
            let token = scope.create_barrier_token(barrier.index().unwrap(), level);
            scope.register(Instruction::new(BarrierOps::Arrive { barrier }, *token));
            token.into()
        })
    }

    /// Arrive at the barrier, decrementing arrival count. Additionally increments expected count.
    #[allow(unused_variables)]
    pub fn arrive_and_expect_tx(&self, arrival_count: u32, transaction_count: u32) -> BarrierToken {
        intrinsic!(|scope| {
            let barrier = *self.expand;
            let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type() else {
                unreachable!()
            };
            let token = scope.create_barrier_token(barrier.index().unwrap(), level);
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
        })
    }

    /// Increments the expected count of the barrier.
    #[allow(unused_variables)]
    pub fn expect_tx(&self, expected_count: u32) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
            let transaction_count: ExpandElement = expected_count.into();
            scope.register(BarrierOps::ExpectTx {
                barrier,
                transaction_count_update: transaction_count.consume(),
            });
        })
    }

    /// Wait until all data is loaded
    pub fn arrive_and_wait(&self) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
            scope.register(BarrierOps::ArriveAndWait { barrier });
        })
    }

    /// Wait at the barrier until all arrivals are done
    #[allow(unused_variables)]
    pub fn wait(&self, token: BarrierToken) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
            let token = *token.expand;
            scope.register(BarrierOps::Wait { barrier, token });
        })
    }

    /// Wait at the barrier until the `phase` is completed. Doesn't require a token, but needs phase
    /// to be managed manually.
    #[allow(unused_variables)]
    pub fn wait_parity(&self, phase: u32) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
            let phase = *phase.expand;
            scope.register(BarrierOps::WaitParity { barrier, phase });
        })
    }
}

// Copy async

/// Copy the source slice in global memory to destination in shared memory with a low level async
/// copy. This only copies up to 128 bits/16 bytes, and does not synchronize. Use
/// `barrier.copy_async_arrive` to make the reads visible.
/// `copy_size` is in terms of elements to simplify copying between different line sizes.
///
/// # Safety
///
/// This will try to copy the entire `copy_size`, so make sure the full width is in bounds.
/// Starting address must be aligned to the full copy size.
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
/// Will only copy the length of the source slice, and zero fill the rest. Source length must be
/// <= copy size.
///
/// # Safety
/// Starting address must be aligned to the full copy size.
/// **This will silently fail if the address is only aligned to the source length and not the copy size!**
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

#[cube(self_type = "ref")]
impl Barrier {
    /// Makes all previous `copy_async` operations visible on the barrier.
    /// Should be called once after all copies have been dispatched, before reading from the shared
    /// memory.
    ///
    /// Does *not* count as an arrive in terms of the barrier arrival count. So `arrive` or
    /// `arrive_and_wait` should still be called afterwards.
    pub fn commit_copy_async(&self) {
        intrinsic!(|scope| {
            let barrier = *self.expand;
            let StorageType::Opaque(OpaqueType::Barrier(level)) = barrier.ty.storage_type() else {
                unreachable!()
            };
            let token = scope.create_barrier_token(barrier.index().unwrap(), level);
            scope.register(Instruction::new(
                BarrierOps::CommitCopyAsync { barrier },
                *token,
            ));
        })
    }
}

impl Deref for Shared<Barrier> {
    type Target = Barrier;

    fn deref(&self) -> &Self::Target {
        unexpanded!()
    }
}
impl Deref for SharedExpand<Barrier> {
    type Target = BarrierExpand;

    fn deref(&self) -> &Self::Target {
        unsafe { self.as_type_ref_unchecked::<Barrier>() }
    }
}

impl DerefMut for Shared<Barrier> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        todo!()
    }
}
impl DerefMut for SharedExpand<Barrier> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.as_type_mut_unchecked::<Barrier>() }
    }
}

impl From<SharedExpand<Barrier>> for BarrierExpand {
    fn from(value: SharedExpand<Barrier>) -> Self {
        value.expand.into()
    }
}
