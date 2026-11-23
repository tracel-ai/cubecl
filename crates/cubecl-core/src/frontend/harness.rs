//! The values a cube function's `harness` hands back to the host, see
//! `cubecl-macros/src/generate/harness.rs`.

use alloc::{boxed::Box, vec::Vec};
use cubecl_common::bytes::Bytes;
use cubecl_ir::ContextExt;
use cubecl_runtime::{kernel::KernelDefinition, server::Handle};
use variadics_please::all_tuples;

use crate::prelude::*;
use crate::{self as cubecl};

/// A value a harness kernel can return to the host.
///
/// Implemented for every scalar, for arrays and vectors of them, for `bool`
/// and vectors of it, for tuples of them, and for structs with
/// `#[derive(CubeHarness)]`.
///
/// An array's length and a vector's width are comptime, so they're only
/// known once the kernel is expanded. A harness returning one expands its
/// kernel once before allocating, and [`__expand_store`](Self::__expand_store)
/// records each length on the way, which [`alloc`](Self::alloc) and
/// [`read`](Self::read) take back in the same order. Values without an array
/// or a vector ignore them.
pub trait HarnessOutput: CubeType + Sized {
    /// The buffers a kernel stores the value into.
    type Out: LaunchArg;

    /// The value the host gets back: `Self`, except for an array, which comes
    /// back as a `Vec`.
    type Host;

    /// Allocate the buffers, pushing each one's handle in the order [`read`](Self::read) takes
    /// them back.
    fn alloc(
        client: &Client,
        lens: &mut impl Iterator<Item = usize>,
        handles: &mut Vec<Handle>,
    ) -> <Self::Out as LaunchArg>::RuntimeArg;

    /// Store the value into its buffers.
    fn __expand_store(
        scope: &Scope,
        out: &mut <Self::Out as CubeType>::ExpandType,
        value: Self::ExpandType,
    );

    /// Rebuild the value from the bytes of its buffers.
    fn read(
        bytes: &mut impl Iterator<Item = Bytes>,
        lens: &mut impl Iterator<Item = usize>,
    ) -> Self::Host;
}

/// The lengths of the arrays a harness kernel returns, in store order, kept
/// in the kernel's context while it's expanded.
#[derive(Default)]
struct HarnessLens(Vec<usize>);

/// Store a harness kernel's return value, recording its arrays' lengths.
#[doc(hidden)]
pub fn __expand_store_output<T: HarnessOutput>(
    scope: &Scope,
    out: &mut <T::Out as CubeType>::ExpandType,
    value: T::ExpandType,
) {
    scope.ctx_mut().set_aux_ty(HarnessLens::default());
    T::__expand_store(scope, out, value);
}

/// The lengths of the arrays an expanded harness kernel returns.
#[doc(hidden)]
pub fn __output_lens(definition: &KernelDefinition) -> Vec<usize> {
    definition.body.ctx().aux_ty::<HarnessLens>().0.clone()
}

#[cube]
fn store_scalar<T: CubePrimitive>(out: &mut Box<[T]>, value: T) {
    out[0] = value;
}

impl<T: CubePrimitive + CubeElement> HarnessOutput for T {
    type Out = Box<[T]>;
    type Host = T;

    fn alloc(
        client: &Client,
        _lens: &mut impl Iterator<Item = usize>,
        handles: &mut Vec<Handle>,
    ) -> BufferArg {
        let handle = client.empty(core::mem::size_of::<T>());
        handles.push(handle.clone());
        // SAFETY: the buffer holds exactly one element.
        unsafe { BufferArg::from_raw_parts(handle, 1) }
    }

    fn __expand_store(scope: &Scope, out: &mut NativeExpand<Box<[T]>>, value: NativeExpand<T>) {
        store_scalar::expand::<T>(scope, out, value);
    }

    fn read(bytes: &mut impl Iterator<Item = Bytes>, _lens: &mut impl Iterator<Item = usize>) -> T {
        let bytes = bytes.next().expect("a harness buffer for every scalar");
        T::from_bytes(&bytes)[0]
    }
}

#[cube]
fn store_array<T: CubePrimitive>(out: &mut Box<[T]>, value: Array<T>) {
    for i in 0..value.len() {
        out[i] = value[i];
    }
}

/// An array comes back as a `Vec` of its length.
impl<T: CubePrimitive + CubeElement> HarnessOutput for Array<T> {
    type Out = Box<[T]>;
    type Host = Vec<T>;

    fn alloc(
        client: &Client,
        lens: &mut impl Iterator<Item = usize>,
        handles: &mut Vec<Handle>,
    ) -> BufferArg {
        let len = lens.next().expect("a harness length for every array");
        // A buffer can't be empty, so an empty array still gets an element.
        let capacity = len.max(1);
        let handle = client.empty(capacity * core::mem::size_of::<T>());
        handles.push(handle.clone());
        // SAFETY: the buffer holds exactly `capacity` elements.
        unsafe { BufferArg::from_raw_parts(handle, capacity) }
    }

    fn __expand_store(
        scope: &Scope,
        out: &mut NativeExpand<Box<[T]>>,
        value: NativeExpand<Array<T>>,
    ) {
        let len = Array::<T>::__expand_len(scope, &value);
        scope.ctx_mut().aux_ty_mut::<HarnessLens>().0.push(len);
        store_array::expand::<T>(scope, out, value);
    }

    fn read(
        bytes: &mut impl Iterator<Item = Bytes>,
        lens: &mut impl Iterator<Item = usize>,
    ) -> Vec<T> {
        let bytes = bytes.next().expect("a harness buffer for every array");
        let len = lens.next().expect("a harness length for every array");
        T::from_bytes(&bytes)[..len].to_vec()
    }
}

#[cube]
fn store_vector<T: Scalar, N: Size>(out: &mut Box<[T]>, value: Vector<T, N>) {
    #[unroll]
    for i in 0..N::value() {
        out[i] = value.extract(i);
    }
}

/// A vector comes back as a `Vec` of its width, which, like an array's
/// length, is only known once the kernel is expanded.
impl<T: Scalar + CubeElement, N: Size> HarnessOutput for Vector<T, N> {
    type Out = Box<[T]>;
    type Host = Vec<T>;

    fn alloc(
        client: &Client,
        lens: &mut impl Iterator<Item = usize>,
        handles: &mut Vec<Handle>,
    ) -> BufferArg {
        <Array<T> as HarnessOutput>::alloc(client, lens, handles)
    }

    fn __expand_store(
        scope: &Scope,
        out: &mut NativeExpand<Box<[T]>>,
        value: NativeExpand<Vector<T, N>>,
    ) {
        let width = N::__expand_value(scope);
        scope.ctx_mut().aux_ty_mut::<HarnessLens>().0.push(width);
        store_vector::expand::<T, N>(scope, out, value);
    }

    fn read(
        bytes: &mut impl Iterator<Item = Bytes>,
        lens: &mut impl Iterator<Item = usize>,
    ) -> Vec<T> {
        <Array<T> as HarnessOutput>::read(bytes, lens)
    }
}

#[cube]
fn store_bool(out: &mut Box<[u32]>, value: bool) {
    out[0] = u32::cast_from(value);
}

/// A truth has no buffer of its own: it is stored as a `u32`, one where it holds.
impl HarnessOutput for bool {
    type Out = Box<[u32]>;
    type Host = bool;

    fn alloc(
        client: &Client,
        lens: &mut impl Iterator<Item = usize>,
        handles: &mut Vec<Handle>,
    ) -> BufferArg {
        <u32 as HarnessOutput>::alloc(client, lens, handles)
    }

    fn __expand_store(
        scope: &Scope,
        out: &mut NativeExpand<Box<[u32]>>,
        value: NativeExpand<bool>,
    ) {
        store_bool::expand(scope, out, value);
    }

    fn read(
        bytes: &mut impl Iterator<Item = Bytes>,
        lens: &mut impl Iterator<Item = usize>,
    ) -> bool {
        <u32 as HarnessOutput>::read(bytes, lens) != 0
    }
}

#[cube]
fn store_bool_vector<N: Size>(out: &mut Box<[u32]>, value: Vector<bool, N>) {
    #[unroll]
    for i in 0..N::value() {
        out[i] = u32::cast_from(value.extract(i));
    }
}

/// A vector of truths comes back as a `Vec<bool>` of its width, each lane stored as a `u32`.
impl<N: Size> HarnessOutput for Vector<bool, N> {
    type Out = Box<[u32]>;
    type Host = Vec<bool>;

    fn alloc(
        client: &Client,
        lens: &mut impl Iterator<Item = usize>,
        handles: &mut Vec<Handle>,
    ) -> BufferArg {
        <Array<u32> as HarnessOutput>::alloc(client, lens, handles)
    }

    fn __expand_store(
        scope: &Scope,
        out: &mut NativeExpand<Box<[u32]>>,
        value: NativeExpand<Vector<bool, N>>,
    ) {
        let width = N::__expand_value(scope);
        scope.ctx_mut().aux_ty_mut::<HarnessLens>().0.push(width);
        store_bool_vector::expand::<N>(scope, out, value);
    }

    fn read(
        bytes: &mut impl Iterator<Item = Bytes>,
        lens: &mut impl Iterator<Item = usize>,
    ) -> Vec<bool> {
        <Array<u32> as HarnessOutput>::read(bytes, lens)
            .into_iter()
            .map(|lane| lane != 0)
            .collect()
    }
}

/// Nothing to hand back, for a function that only writes into its slices.
impl HarnessOutput for () {
    type Out = ();
    type Host = ();

    fn alloc(
        _client: &Client,
        _lens: &mut impl Iterator<Item = usize>,
        _handles: &mut Vec<Handle>,
    ) {
    }

    fn __expand_store(_scope: &Scope, _out: &mut (), _value: ()) {}

    fn read(_bytes: &mut impl Iterator<Item = Bytes>, _lens: &mut impl Iterator<Item = usize>) {}
}

macro_rules! harness_tuple {
    ($(($T:ident, $t:ident)),*) => {
        impl<$($T: HarnessOutput),*> HarnessOutput for ($($T,)*) {
            type Out = ($($T::Out,)*);
            type Host = ($($T::Host,)*);

            fn alloc(
                client: &Client,
                lens: &mut impl Iterator<Item = usize>,
                handles: &mut Vec<Handle>,
            ) -> <Self::Out as LaunchArg>::RuntimeArg {
                ($($T::alloc(client, lens, handles),)*)
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

            fn read(
                bytes: &mut impl Iterator<Item = Bytes>,
                lens: &mut impl Iterator<Item = usize>,
            ) -> Self::Host {
                ($($T::read(bytes, lens),)*)
            }
        }
    };
}

all_tuples!(harness_tuple, 1, 12, T, t);
