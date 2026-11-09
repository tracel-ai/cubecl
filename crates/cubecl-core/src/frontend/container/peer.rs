//! Buffers in memory shared with the other devices of a runtime.

use crate::{
    frontend::container::{buffer_len::expand_buffer_length_native, slice},
    prelude::*,
    unexpanded,
};
use core::marker::PhantomData;
use cubecl_runtime::server::Handle;
use serde::{Deserialize, Serialize};

/// A buffer in memory a client [shared with its peers](cubecl_runtime::client::Client::share_with_peers),
/// as a kernel parameter: `&Peer<[T]>` reads it, `&mut Peer<[T]>` writes it.
///
/// Kernels on other devices may read and write the buffer while this one runs, wherever the
/// memory lives: on the launching device, where its owner's kernels bind it, or on another one,
/// which kernels there bind directly. Inside the kernel it is a slice like any other, with what
/// sharing changes:
///
/// - The compiler promises nothing about who writes it or what it aliases, so a read is never
///   served from a cache another device's writes go around. The kernel's other buffers keep
///   every promise.
/// - An atomic on it synchronizes at
///   [`SyncScope::System`](crate::ir::dialect::synchronization::SyncScope::System), whatever
///   scope the call names: a device-scope atomic may compile to an instruction that is not
///   atomic across the link between the devices.
///
/// Nothing orders the accesses of different devices. Kernels hand data off through an atomic flag
/// in a peer buffer, with [`sync_system()`](crate::prelude::sync_system()) between the data and the flag on both sides.
pub struct Peer<T: ?Sized> {
    _buffer: PhantomData<T>,
}

impl<C: CubePrimitive> CubeType for Peer<[C]> {
    type ExpandType = SliceExpand<C>;
}

/// A `Peer` buffer reads as a slice inside a kernel. Only the expansion runs: these exist for
/// the kernel's body to type-check as Rust.
impl<C: CubePrimitive> core::ops::Deref for Peer<[C]> {
    type Target = [C];

    fn deref(&self) -> &[C] {
        unexpanded!()
    }
}

impl<C: CubePrimitive> core::ops::DerefMut for Peer<[C]> {
    fn deref_mut(&mut self) -> &mut [C] {
        unexpanded!()
    }
}

/// The compilation argument of a [`Peer`] buffer: it never aliases another binding, since its
/// memory is shared and an alias would be shared by accident.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct PeerCompilationArg;

/// A shared buffer passed to a [`Peer`] parameter.
pub struct PeerArg {
    pub(crate) handle: BufferBinding,
}

impl PeerArg {
    /// `handle`, of `length` elements, for a [`Peer`] parameter.
    ///
    /// # Safety
    ///
    /// Specifying the wrong length may lead to out-of-bounds reads and writes.
    ///
    /// # Panics
    ///
    /// When `handle` was never [shared](cubecl_runtime::client::Client::share_with_peers): the
    /// memory of a buffer only its device knows cannot be bound as shared.
    pub unsafe fn from_raw_parts(handle: &Handle, length: usize) -> Self {
        assert!(
            handle.peer.is_some(),
            "a `Peer` parameter takes a handle shared with `share_with_peers`"
        );
        Self {
            handle: unsafe { BufferBinding::from_raw_parts(handle.clone(), length) },
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Peer<[C]> {
    type RuntimeArg = PeerArg;
    type CompilationArg = PeerCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        let elem_size = launcher.with_scope(|scope| C::__expand_size(scope));
        launcher.register_peer(arg, elem_size);
        PeerCompilationArg
    }

    fn expand(_arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> SliceExpand<C> {
        let buffer = builder.peer_buffer(C::__expand_as_type(&builder.scope));
        let scope = &builder.scope;
        let len = expand_buffer_length_native(scope, buffer);
        let slice_var =
            slice::from_raw_parts::<C>(scope, buffer, 0usize.into_expand(scope), len.into());
        slice_var.expand.into()
    }
}
