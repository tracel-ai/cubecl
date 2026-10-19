//! The values a cube function's `harness` hands back to the host, see
//! `cubecl-macros/src/generate/harness.rs`.

use alloc::{boxed::Box, vec::Vec};
use cubecl_common::bytes::Bytes;
use cubecl_runtime::server::Handle;
use variadics_please::all_tuples;

use crate::prelude::*;
use crate::{self as cubecl};

/// A value a harness kernel can return to the host.
///
/// Implemented for every scalar, for tuples of them, and for structs with
/// `#[derive(CubeHarness)]`.
pub trait HarnessOutput: CubeType + Sized {
    /// The buffers a kernel stores the value into.
    type Out: LaunchArg;

    /// Allocate the buffers, pushing each one's handle in the order [`read`](Self::read) takes
    /// them back.
    fn alloc(client: &Client, handles: &mut Vec<Handle>) -> <Self::Out as LaunchArg>::RuntimeArg;

    /// Store the value into its buffers.
    fn __expand_store(
        scope: &Scope,
        out: &mut <Self::Out as CubeType>::ExpandType,
        value: Self::ExpandType,
    );

    /// Rebuild the value from the bytes of its buffers.
    fn read(bytes: &mut impl Iterator<Item = Bytes>) -> Self;
}

#[cube]
fn store_scalar<T: CubePrimitive>(out: &mut Box<[T]>, value: T) {
    out[0] = value;
}

impl<T: CubePrimitive + CubeElement> HarnessOutput for T {
    type Out = Box<[T]>;

    fn alloc(client: &Client, handles: &mut Vec<Handle>) -> BufferArg {
        let handle = client.empty(core::mem::size_of::<T>());
        handles.push(handle.clone());
        // SAFETY: the buffer holds exactly one element.
        unsafe { BufferArg::from_raw_parts(handle, 1) }
    }

    fn __expand_store(scope: &Scope, out: &mut NativeExpand<Box<[T]>>, value: NativeExpand<T>) {
        store_scalar::expand::<T>(scope, out, value);
    }

    fn read(bytes: &mut impl Iterator<Item = Bytes>) -> Self {
        let bytes = bytes.next().expect("a harness buffer for every scalar");
        T::from_bytes(&bytes)[0]
    }
}

macro_rules! harness_tuple {
    ($(($T:ident, $t:ident)),*) => {
        impl<$($T: HarnessOutput),*> HarnessOutput for ($($T,)*) {
            type Out = ($($T::Out,)*);

            fn alloc(client: &Client, handles: &mut Vec<Handle>) -> <Self::Out as LaunchArg>::RuntimeArg {
                ($($T::alloc(client, handles),)*)
            }

            #[allow(non_snake_case)]
            fn __expand_store(
                scope: &Scope,
                out: &mut <Self::Out as CubeType>::ExpandType,
                value: Self::ExpandType,
            ) {
                let ($($T,)*) = out;
                let ($($t,)*) = value;
                $(<$T as HarnessOutput>::__expand_store(scope, $T, $t);)*
            }

            fn read(bytes: &mut impl Iterator<Item = Bytes>) -> Self {
                ($($T::read(bytes),)*)
            }
        }
    };
}

all_tuples!(harness_tuple, 1, 12, T, t);
