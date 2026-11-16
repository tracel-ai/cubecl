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
    handle: cubecl_runtime::server::BufferBinding,
    length: [usize; 1],
    extent: BufferExtent,
}

/// How much of its handle a kernel sees through a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BufferExtent {
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
    /// A packed type is one scalar: a buffer of `e2m1x2` counts its pairs, one byte each.
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

    /// The handle, and the length the kernel sees in its own elements of `elem_size` bytes, each
    /// a vector of `vector_size` scalars.
    ///
    /// # Panics
    ///
    /// When the handle is too small to hold the declared length. The caller broke the contract
    /// of the `unsafe` constructor, so the launch stops there instead of queuing an error.
    pub fn into_kernel_binding(
        self,
        elem_size: usize,
        vector_size: usize,
    ) -> (cubecl_runtime::server::BufferBinding, u64) {
        let length = self.kernel_len(elem_size, vector_size);
        (self.handle, length)
    }

    /// The length the kernel sees, in its own elements.
    fn kernel_len(&self, elem_size: usize, vector_size: usize) -> u64 {
        match self.extent {
            BufferExtent::Handle => self.handle.size_in_used() / elem_size as u64,
            BufferExtent::Declared => {
                self.declared_kernel_len(elem_size / vector_size, vector_size)
            }
        }
    }

    /// The whole vectors of `vector_size` scalars of `scalar_size` bytes the declared length
    /// holds, once checked against the handle.
    fn declared_kernel_len(&self, scalar_size: usize, vector_size: usize) -> u64 {
        let length = self.length[0] as u64;
        let scalar_size = scalar_size as u64;
        let size = self.handle.size_in_used();
        assert!(
            length * scalar_size <= size,
            "A buffer declared with {length} scalars is bound to a handle of {size} bytes, which \
             holds {} scalars",
            size / scalar_size,
        );
        length / vector_size as u64
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

/// A tensor's memory as a buffer of its shape's product scalars. The kernel sees the whole handle,
/// see [`BufferExtent::Handle`].
impl From<TensorBinding> for BufferBinding {
    fn from(tensor: TensorBinding) -> Self {
        let length = tensor.size();
        Self {
            handle: tensor.handle,
            length: [length],
            extent: BufferExtent::Handle,
        }
    }
}
