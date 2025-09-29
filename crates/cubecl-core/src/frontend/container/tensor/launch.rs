use std::marker::PhantomData;

use serde::{Deserialize, Serialize};

use crate::{
    Runtime,
    compute::{KernelBuilder, KernelLauncher},
    ir::{Id, LineSize, Type},
    prelude::{
        ArgSettings, CompilationArg, CubePrimitive, ExpandElementTyped, LaunchArg, LaunchArgExpand,
    },
};

use super::Tensor;

/// Errors that can occur when constructing a tensor handle safely.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorHandleError {
    /// Rank of shape and strides differ.
    RankMismatch {
        shape_rank: usize,
        stride_rank: usize,
    },
    /// Element size must be > 0.
    ElemSizeZero,
    /// A stride is zero for a dimension with extent > 1.
    ZeroStride { axis: usize },
}

/// Errors that can occur when converting a handle to a runtime tensor argument.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorArgError {
    /// Requested vectorization factor is not supported by the runtime.
    UnsupportedVectorization {
        requested: u8,
        supported: &'static [u8],
    },
    /// Inner-most dimension is not contiguous (stride != 1) while vectorization > 1.
    NonContiguousInner,
    /// Inner-most dimension is not divisible by the vectorization factor.
    MisalignedVectorization { last_dim: usize, factor: u8 },
}

/// Argument to be used for [tensors](Tensor) passed as arguments to kernels.
#[derive(Debug)]
pub enum TensorArg<'a, R: Runtime> {
    /// The tensor is passed with a tensor handle.
    Handle {
        /// The tensor handle.
        handle: TensorHandleRef<'a, R>,
        /// The vectorization factor.
        line_size: u8,
    },
    /// The tensor is aliasing another input tensor.
    Alias {
        /// The position of the input tensor.
        input_pos: usize,
    },
}

/// Tensor representation with a reference to the [server handle](cubecl_runtime::server::Handle),
/// the strides and the shape.
pub struct TensorHandleRef<'a, R: Runtime> {
    pub handle: &'a cubecl_runtime::server::Handle,
    pub strides: &'a [usize],
    pub shape: &'a [usize],
    pub elem_size: usize,
    pub runtime: PhantomData<R>,
}

impl<'a, R: Runtime> Clone for TensorHandleRef<'a, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, R: Runtime> Copy for TensorHandleRef<'a, R> {}

impl<R: Runtime> TensorHandleRef<'_, R> {
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

impl<R: Runtime> core::fmt::Debug for TensorHandleRef<'_, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "TensorHandleRef {{ strides: {:?}, shape: {:?} }}",
            self.strides, self.shape
        )
    }
}

/// Compilation argument for a [tensor](Tensor).
#[derive(Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct TensorCompilationArg {
    pub inplace: Option<Id>,
    pub line_size: LineSize,
}

impl CompilationArg for TensorCompilationArg {}

impl<C: CubePrimitive> LaunchArgExpand for Tensor<C> {
    type CompilationArg = TensorCompilationArg;

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        builder
            .input_tensor(Type::new(C::as_type(&builder.scope)).line(arg.line_size))
            .into()
    }
    fn expand_output(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> ExpandElementTyped<Tensor<C>> {
        match arg.inplace {
            Some(id) => builder.inplace_output(id).into(),
            None => builder
                .output_tensor(Type::new(C::as_type(&builder.scope)).line(arg.line_size))
                .into(),
        }
    }
}

impl<C: CubePrimitive> LaunchArg for Tensor<C> {
    type RuntimeArg<'a, R: Runtime> = TensorArg<'a, R>;

    fn compilation_arg<R: Runtime>(runtime_arg: &Self::RuntimeArg<'_, R>) -> Self::CompilationArg {
        match runtime_arg {
            TensorArg::Handle { line_size, .. } => TensorCompilationArg {
                inplace: None,
                line_size: *line_size as u32,
            },
            TensorArg::Alias { input_pos } => TensorCompilationArg {
                inplace: Some(*input_pos as Id),
                line_size: 0,
            },
        }
    }
}

impl<'a, R: Runtime> TensorArg<'a, R> {
    /// Create a new tensor argument specified with its vectorization factor.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts<E: CubePrimitive>(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
        factor: u8,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorHandleRef::from_raw_parts(
                    handle,
                    strides,
                    shape,
                    E::size().expect("Element should have a size"),
                ),
                line_size: factor,
            }
        }
    }

    /// Create a new tensor argument specified with its vectorization factor with a manual element
    /// size in bytes.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bound reads and writes.
    pub unsafe fn from_raw_parts_and_size(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
        factor: u8,
        elem_size: usize,
    ) -> Self {
        unsafe {
            Self::Handle {
                handle: TensorHandleRef::from_raw_parts(handle, strides, shape, elem_size),
                line_size: factor,
            }
        }
    }

    /// Create an alias argument.
    pub fn alias(position: usize) -> Self {
        Self::Alias {
            input_pos: position,
        }
    }
}

impl<R: Runtime> ArgSettings<R> for TensorArg<'_, R> {
    fn register(&self, launcher: &mut KernelLauncher<R>) {
        launcher.register_tensor(self);
    }
}

impl<'a, R: Runtime> TensorHandleRef<'a, R> {
    /// Convert the handle into a [tensor argument](TensorArg).
    pub fn as_tensor_arg(&'a self, vectorization: u8) -> TensorArg<'a, R> {
        // In debug builds, assert that the requested vectorization is supported
        // by the runtime. Validation of the chosen factor should normally be
        // performed upstream (at selection time) to avoid redundant checks in
        // hot paths.
        debug_assert!(
            R::supported_line_sizes().contains(&vectorization),
            "unsupported vectorization {} (supported: {:?})",
            vectorization,
            R::supported_line_sizes()
        );
        unsafe {
            TensorArg::from_raw_parts_and_size(
                self.handle,
                self.strides,
                self.shape,
                vectorization,
                self.elem_size,
            )
        }
    }
    /// Convert the handle into a [tensor argument](TensorArg) with basic safety checks
    /// for vectorization compatibility.
    ///
    /// Note: This convenience is primarily intended for host wrappers / FFI
    /// ingestion paths. In internal code, prefer validating the chosen
    /// vectorization factor at selection time and then calling
    /// [`as_tensor_arg`], to avoid redundant work in hot paths.
    ///
    /// This does not enforce inner‑most contiguity or alignment requirements as
    /// kernels may vectorize along axes other than the innermost.
    pub fn try_as_tensor_arg(
        &'a self,
        vectorization: u8,
    ) -> Result<TensorArg<'a, R>, TensorArgError> {
        if !R::supported_line_sizes().contains(&vectorization) {
            return Err(TensorArgError::UnsupportedVectorization {
                requested: vectorization,
                supported: R::supported_line_sizes(),
            });
        }
        Ok(self.as_tensor_arg(vectorization))
    }

    /// Create a handle from raw parts.
    ///
    /// # Safety
    ///
    /// If you provide wrong strides or shapes, it might create undefined behavior caused by
    /// out-of-bounds reads and writes.
    pub unsafe fn from_raw_parts(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
        elem_size: usize,
    ) -> Self {
        // Basic invariants for debug builds only; upstream layers are expected
        // to ensure correctness in release builds.
        debug_assert_eq!(
            shape.len(),
            strides.len(),
            "rank mismatch (shape={}, strides={})",
            shape.len(),
            strides.len()
        );
        debug_assert!(elem_size > 0, "element size must be > 0");
        // Note: zero strides are permitted here to support explicit broadcast
        // views in advanced/internal paths. The checked constructor
        // (`try_from_parts`) rejects them when `d > 1` to provide safety at
        // boundaries; callers who intentionally need zero‑stride broadcasting
        // can opt into this `unsafe` API.
        Self {
            handle,
            strides,
            shape,
            elem_size,
            runtime: PhantomData,
        }
    }

    /// Safely create a tensor handle from raw parts with basic shape/stride validation.
    ///
    /// Note: This is mainly useful for host / FFI boundaries to surface clear
    /// errors early. Internal code should ensure these invariants when
    /// constructing handles and may use the `unsafe` constructor directly in
    /// performance‑critical paths.
    pub fn try_from_parts(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
        elem_size: usize,
    ) -> Result<Self, TensorHandleError> {
        if shape.len() != strides.len() {
            return Err(TensorHandleError::RankMismatch {
                shape_rank: shape.len(),
                stride_rank: strides.len(),
            });
        }
        if elem_size == 0 {
            return Err(TensorHandleError::ElemSizeZero);
        }
        // Disallow zero strides when corresponding dimension extent > 1 (broadcasted dims with extent 1 are allowed).
        for (i, (&s, &d)) in strides.iter().zip(shape.iter()).enumerate() {
            if s == 0 && d > 1 {
                return Err(TensorHandleError::ZeroStride { axis: i });
            }
        }
        Ok(unsafe { Self::from_raw_parts(handle, strides, shape, elem_size) })
    }

    /// Safely create a tensor handle from raw parts using the element type for size.
    pub fn try_from_typed<E: CubePrimitive>(
        handle: &'a cubecl_runtime::server::Handle,
        strides: &'a [usize],
        shape: &'a [usize],
    ) -> Result<Self, TensorHandleError> {
        let elem_size = E::size().expect("Element should have a size");
        Self::try_from_parts(handle, strides, shape, elem_size)
    }
}

impl core::fmt::Display for TensorHandleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorHandleError::RankMismatch {
                shape_rank,
                stride_rank,
            } => {
                write!(
                    f,
                    "rank mismatch (shape={}, strides={})",
                    shape_rank, stride_rank
                )
            }
            TensorHandleError::ElemSizeZero => write!(f, "element size must be > 0"),
            TensorHandleError::ZeroStride { axis } => {
                write!(f, "zero stride on axis {} with extent > 1", axis)
            }
        }
    }
}

impl core::fmt::Display for TensorArgError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            TensorArgError::UnsupportedVectorization {
                requested,
                supported,
            } => {
                write!(
                    f,
                    "unsupported vectorization {}, supported: {:?}",
                    requested, supported
                )
            }
            TensorArgError::NonContiguousInner => write!(
                f,
                "non-contiguous innermost dimension for vectorized access"
            ),
            TensorArgError::MisalignedVectorization { last_dim, factor } => write!(
                f,
                "innermost dimension {} not divisible by vectorization {}",
                last_dim, factor
            ),
        }
    }
}
