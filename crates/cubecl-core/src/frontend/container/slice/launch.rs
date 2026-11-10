use crate::{
    frontend::container::{buffer_len::expand_buffer_length_native, slice},
    prelude::*,
};
use alloc::boxed::Box;
use cubecl_zspace::Tiling;
use serde::{Deserialize, Serialize};

#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct BufferCompilationArg {
    pub inplace: Option<usize>,
}

/// Buffer representation with a reference to the [server handle](cubecl_runtime::server::Handle).
///
/// Its length counts scalars of the storage type, whatever the vector size of the kernel
/// parameter it is bound to.
pub struct BufferBinding {
    pub handle: cubecl_runtime::server::BufferBinding,
    pub(crate) length: [usize; 1],
    pub(crate) extent: BufferExtent,
}

/// How much of its handle a kernel sees through a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BufferExtent {
    /// The declared length, which may stop short of the handle's end.
    Declared,
    /// The whole handle. A tensor viewed as a buffer still reaches its memory through its
    /// strides, so its shape is no bound on where it reads: a broadcast tensor's shape holds
    /// more elements than its memory, a strided one's fewer than it spans.
    Handle,
}

pub enum BufferArg {
    /// The buffer is passed with a buffer handle.
    Handle {
        /// The buffer handle.
        handle: BufferBinding,
    },
    /// The buffer is aliasing another input buffer.
    ///
    /// The kernel reads the length of the buffer it aliases, `length` only answers
    /// [`len`](BufferArg::len).
    Alias {
        /// The position of the input buffer.
        input_pos: usize,
        /// The length of the aliased buffer, in scalars of the storage type.
        length: [usize; 1],
    },
}

impl BufferArg {
    /// Create a new buffer argument.
    ///
    /// The kernel sees `length`, counted in scalars of the storage type: a kernel taking vectors
    /// of `N` scalars sees the `length / N` whole vectors it holds.
    ///
    /// # Safety
    ///
    /// The kernel trusts `length` for its bounds, so `handle` must hold `length` initialized
    /// scalars for the kernel to read them.
    ///
    /// # Panics
    ///
    /// At launch, when `handle` is too small to hold `length` scalars.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe {
            BufferArg::Handle {
                handle: BufferBinding::from_raw_parts(handle, length),
            }
        }
    }
    /// Create a new buffer argument from a binding.
    ///
    /// See [`from_raw_parts`](Self::from_raw_parts) for what `length` means.
    ///
    /// # Safety
    ///
    /// The kernel trusts `length` for its bounds, so `binding` must hold `length` initialized
    /// scalars for the kernel to read them.
    pub unsafe fn from_raw_parts_binding(
        binding: cubecl_runtime::server::BufferBinding,
        length: usize,
    ) -> Self {
        unsafe {
            BufferArg::Handle {
                handle: BufferBinding::from_raw_parts_binding(binding, length),
            }
        }
    }

    /// The buffer's length in scalars of the storage type, as it was declared.
    pub fn len(&self) -> usize {
        match self {
            BufferArg::Handle { handle } => handle.length[0],
            BufferArg::Alias { length, .. } => length[0],
        }
    }

    /// Whether the buffer was declared empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn alias(input_pos: usize, length: usize) -> Self {
        Self::Alias {
            input_pos,
            length: [length],
        }
    }

    pub fn size(&self) -> usize {
        match self {
            BufferArg::Handle { handle } => handle.length[0],
            BufferArg::Alias { length, .. } => length[0],
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            BufferArg::Handle { handle } => &handle.length,
            BufferArg::Alias { length, .. } => length,
        }
    }
}

impl BufferBinding {
    /// Create a new buffer handle reference.
    ///
    /// See [`BufferArg::from_raw_parts`] for what `length` means.
    ///
    /// # Safety
    ///
    /// The kernel trusts `length` for its bounds, so `handle` must hold `length` initialized
    /// scalars for the kernel to read them.
    pub unsafe fn from_raw_parts(handle: cubecl_runtime::server::Handle, length: usize) -> Self {
        unsafe { Self::from_raw_parts_binding(handle.binding(), length) }
    }

    /// Create a new buffer handle reference.
    ///
    /// See [`BufferArg::from_raw_parts`] for what `length` means.
    ///
    /// # Safety
    ///
    /// The kernel trusts `length` for its bounds, so `handle` must hold `length` initialized
    /// scalars for the kernel to read them.
    pub unsafe fn from_raw_parts_binding(
        handle: cubecl_runtime::server::BufferBinding,
        length: usize,
    ) -> Self {
        Self {
            handle,
            length: [length],
            extent: BufferExtent::Declared,
        }
    }

    /// A tensor's memory as a buffer of `length` scalars, its shape's product. The kernel sees
    /// the whole handle, see [`BufferExtent::Handle`].
    pub(crate) fn from_tensor_binding(
        handle: cubecl_runtime::server::BufferBinding,
        length: usize,
    ) -> Self {
        Self {
            handle,
            length: [length],
            extent: BufferExtent::Handle,
        }
    }

    /// Return the handle as a tensor instead of a buffer.
    pub fn into_tensor(self) -> TensorBinding {
        let shape = self.length.into();

        TensorBinding {
            handle: self.handle,
            strides: [1].into(),
            shape,
            tiling: Tiling::UNTILED,
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Box<[C]> {
    type RuntimeArg = BufferArg;
    type CompilationArg = BufferCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        <[C]>::register(arg, launcher)
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<Box<[C]>> {
        <[C]>::expand(arg, builder).expand.into()
    }
}

impl<C: CubePrimitive> LaunchArg for [C] {
    type RuntimeArg = BufferArg;
    type CompilationArg = BufferCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        let (elem_size, vector_size) =
            launcher.with_scope(|scope| (C::__expand_size(scope), C::__expand_vector_size(scope)));
        let inplace = match &arg {
            BufferArg::Handle { .. } => None,
            BufferArg::Alias { input_pos, .. } => Some(*input_pos),
        };
        launcher.register_buffer(arg, elem_size, vector_size);

        BufferCompilationArg { inplace }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> NativeExpand<[C]> {
        let buffer = match arg.inplace {
            Some(id) => builder.inplace(id),
            None => builder.buffer(C::__expand_as_type(&builder.scope)),
        };
        let scope = &builder.scope;
        let len = expand_buffer_length_native(scope, buffer);
        let slice_var =
            slice::from_raw_parts::<C>(scope, buffer, 0usize.into_expand(scope), len.into());
        slice_var.expand.into()
    }
}
