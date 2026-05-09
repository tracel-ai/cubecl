use cubecl::prelude::*;
use cubecl_core::{self as cubecl, ir::UIntKind, zspace::Shape};

use crate::tensor::{
    View, is_contiguous, is_contiguous_pitched,
    launch::{ConcreteLayout, ConcreteLayoutLaunch, MemoryArg, ViewArg, ViewLayoutLaunchArg},
    layout::{
        Coords1d, Layout, LayoutExpand, VirtualLayoutOperationsExpand,
        permuted::{PermutedLayout, PermutedLayoutCompilationArg, PermutedLayoutLaunch},
        plain::PlainLayout,
        strided::{StridedLayout, StridedLayoutCompilationArg},
    },
};

/// Maps a linear index based on vector count to a potentially strided tensor. Only applies the
/// necessary level of striding, either none, only the last dim (for freshly allocated strided
/// tensors), or all dimensions.
///
/// Treats indices as the vector index, with the shape being adjusted for vector size.
///
/// `Layout` version of [`crate::tensor::contiguous::index_offset_contiguous()`]
#[derive(CubeType, Clone)]
pub enum LinearViewLayout {
    /// Input is contiguous, no mapping
    Plain(PlainLayout),
    /// Strided tensor, i.e. freshly allocated but not permuted
    Strided(StridedLayout),
    /// Permuted layout, tracks the entire shape/strides and not just the last dim
    Permuted(PermutedLayout),
}

impl LinearViewLayoutExpand {
    fn __expand_inner_method(
        &self,
        _scope: &Scope,
    ) -> &dyn VirtualLayoutOperationsExpand<Coords1d, Coords1d> {
        match self {
            LinearViewLayoutExpand::Plain(layout) => layout,
            LinearViewLayoutExpand::Strided(layout) => layout,
            LinearViewLayoutExpand::Permuted(layout) => layout,
        }
    }
}

#[derive(Default)]
pub struct LinearViewLayoutLaunch {
    reference_shape: Option<Shape>,
}

impl ViewLayoutLaunchArg for LinearViewLayout {
    type RuntimeArg<R: Runtime> = LinearViewLayoutLaunch;
    type CompilationArg = LinearLayoutCompilationArg;

    fn register<R: Runtime, B: MemoryArg>(
        runtime_arg: Self::RuntimeArg<R>,
        buffer: &B,
        ty: Type,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let shape = buffer.shape();
        match runtime_arg.reference_shape {
            Some(reference_shape) if reference_shape.as_slice() != shape => {
                let arg = PermutedLayoutLaunch::from_reference_shape(reference_shape);
                let comp_arg = PermutedLayout::register(arg, buffer, ty, launcher);
                LinearLayoutCompilationArg::Permuted(comp_arg)
            }
            _ => {
                let strides = buffer.strides();
                if is_contiguous(shape, strides) {
                    PlainLayout::register((), buffer, ty, launcher);
                    LinearLayoutCompilationArg::Plain
                } else if is_contiguous_pitched(shape, strides) {
                    let comp_arg = StridedLayout::register((), buffer, ty, launcher);
                    LinearLayoutCompilationArg::Strided(comp_arg)
                } else {
                    let comp_arg =
                        PermutedLayout::register(Default::default(), buffer, ty, launcher);
                    LinearLayoutCompilationArg::Permuted(comp_arg)
                }
            }
        }
    }
    fn expand(
        compilation_arg: &Self::CompilationArg,
        ty: Type,
        builder: &mut cubecl::prelude::KernelBuilder,
    ) -> <Self as cubecl::prelude::CubeType>::ExpandType {
        match compilation_arg {
            LinearLayoutCompilationArg::Plain => {
                LinearViewLayoutExpand::Plain(PlainLayout::expand(&(), ty, builder))
            }
            LinearLayoutCompilationArg::Strided(arg) => {
                LinearViewLayoutExpand::Strided(StridedLayout::expand(arg, ty, builder))
            }
            LinearLayoutCompilationArg::Permuted(arg) => {
                LinearViewLayoutExpand::Permuted(PermutedLayout::expand(arg, ty, builder))
            }
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum LinearLayoutCompilationArg {
    Plain,
    Strided(StridedLayoutCompilationArg),
    Permuted(PermutedLayoutCompilationArg),
}

impl LinearViewLayoutLaunch {
    /// Construct a linear layout from shapes, strides and vector size of the tensor
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a possibly broadcast linear layout from shapes/strides and a reference shape
    pub fn from_reference_shape(reference_shape: Shape) -> Self {
        Self {
            reference_shape: Some(reference_shape),
        }
    }

    /// Construct a possibly broadcast linear layout from a tensor handle and reference handle
    pub fn from_reference_handle<R: Runtime>(reference: TensorBinding<R>) -> Self {
        Self::from_reference_shape(reference.shape)
    }
}

#[cube]
impl Layout for LinearViewLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    #[allow(unused)]
    fn to_source_pos(&self, pos: Self::Coordinates) -> usize {
        intrinsic!(|scope| {
            let inner = self.__expand_inner_method(scope);
            inner.__expand_to_source_pos_virt_method(scope, pos)
        })
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (usize, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        intrinsic!(|scope| {
            let inner = self.__expand_inner_method(scope);
            inner.__expand_shape_virt_method(scope)
        })
    }

    #[allow(unused)]
    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        intrinsic!(|scope| {
            let inner = self.__expand_inner_method(scope);
            inner.__expand_is_in_bounds_virt_method(scope, pos)
        })
    }
}

/// Concrete version of the layout, so it can be launched on its own
pub type LinearLayout = ConcreteLayout<LinearViewLayout>;
pub type LinearLayoutLaunch<R> = ConcreteLayoutLaunch<LinearViewLayout, R>;

/// [`View`] with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type LinearView<E, IO = ReadOnly> = View<E, Coords1d, IO>;
/// Launch type for [`LinearView`].
pub type LinearViewLaunch<R> = ViewArg<Coords1d, R>;

/// Create a linear layout from a handle and vector size
pub fn linear_layout<R: Runtime>(
    handle: &TensorBinding<R>,
    vector_size: VectorSize,
) -> LinearLayoutLaunch<R> {
    LinearLayoutLaunch::from_handle(
        handle,
        // Don't care about type size, only vector size
        Type::new(UIntKind::U32.into()).with_vector_size(vector_size),
        LinearViewLayoutLaunch::new(),
    )
}

/// Create a linear tensor view from a handle
pub fn linear_view<R: Runtime>(handle: TensorBinding<R>) -> LinearViewLaunch<R> {
    let layout = LinearViewLayoutLaunch::new();
    LinearViewLaunch::new_tensor::<LinearViewLayout>(handle.into_tensor_arg(), layout)
}

/// Create a possibly broadcast linear tensor view from a handle and reference handle
pub fn linear_view_with_reference<R: Runtime>(
    handle: TensorBinding<R>,
    reference: TensorBinding<R>,
) -> LinearViewLaunch<R> {
    let layout = LinearViewLayoutLaunch::from_reference_handle(reference);
    LinearViewLaunch::new_tensor::<LinearViewLayout>(handle.into_tensor_arg(), layout)
}

pub fn linear_view_alias<R: Runtime>(handle: &TensorBinding<R>, pos: usize) -> LinearViewLaunch<R> {
    let layout = LinearViewLayoutLaunch::new();
    LinearViewLaunch::new_tensor::<LinearViewLayout>(handle.as_alias(pos), layout)
}
