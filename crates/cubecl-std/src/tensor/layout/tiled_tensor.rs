use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

use crate::tensor::{
    View,
    launch::{ConcreteLayout, ConcreteLayoutLaunch, MemoryArg, ViewArg, ViewLayoutLaunchArg},
    layout::{Coords1d, CoordsDyn, Layout, LayoutExpand, tiled::TiledLayout},
};

/// Composed layout that maps the *semantic* coordinates of a tiled tensor
/// (rank R) to a 1D buffer offset, by first rank-expanding to the *physical*
/// (rank R + n) tiled coordinates, then applying per-axis strides.
///
/// Auto-derives shape, strides, and tile spec at launch from the tensor
/// binding's [`Metadata`](cubecl_core::zspace::metadata::Metadata) — the
/// binding must carry a [`Tiler`](cubecl_core::zspace::metadata::Tiler).
#[derive(CubeType, Clone)]
pub struct TiledTensorLayout {
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    tile_shape: CoordsDyn,
}

#[cube]
impl Layout for TiledTensorLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        // Build a TiledLayout view over `physical_shape` to do semantic -> physical.
        let tiled = TiledLayout::new(
            self.physical_shape.clone(),
            self.start_axis,
            self.tile_shape.clone(),
        );
        let physical = tiled.to_source_pos(pos);

        // Physical (rank R + n) -> buffer offset via per-axis strides.
        let mut offset = 0u32;
        #[unroll]
        for i in 0..self.physical_strides.len() {
            offset += physical[i] * self.physical_strides[i];
        }
        offset as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        let tiled = TiledLayout::new(
            self.physical_shape.clone(),
            self.start_axis,
            self.tile_shape.clone(),
        );
        tiled.shape()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut is_valid = true;
        #[unroll]
        for i in 0..bounds.len() {
            is_valid = is_valid && pos[i] < bounds[i];
        }
        is_valid
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct TiledTensorLayoutCompilationArg {
    physical_shape: <CoordsDyn as LaunchArg>::CompilationArg,
    physical_strides: <CoordsDyn as LaunchArg>::CompilationArg,
    start_axis: u8,
    tile_shape: <CoordsDyn as LaunchArg>::CompilationArg,
}

impl ViewLayoutLaunchArg for TiledTensorLayout {
    type RuntimeArg<R: Runtime> = ();
    type CompilationArg = TiledTensorLayoutCompilationArg;

    fn register<R: Runtime, B: MemoryArg>(
        _: Self::RuntimeArg<R>,
        buffer: &B,
        _ty: Type,
        launcher: &mut KernelLauncher<R>,
    ) -> Self::CompilationArg {
        let tiler = buffer.tiler().expect(
            "TiledTensorLayout requires a buffer carrying a Tiler — \
             rank-expand the tensor's metadata via `Metadata::to_tiled` first",
        );
        let shape = buffer.shape();
        let strides = buffer.strides();

        let shape_arg: <CoordsDyn as LaunchArg>::RuntimeArg<R> =
            shape.iter().map(|&s| s as u32).collect();
        let strides_arg: <CoordsDyn as LaunchArg>::RuntimeArg<R> =
            strides.iter().map(|&s| s as u32).collect();
        let tile_arg: <CoordsDyn as LaunchArg>::RuntimeArg<R> =
            tiler.tile_size.iter().map(|&s| s as u32).collect();

        TiledTensorLayoutCompilationArg {
            physical_shape: <CoordsDyn as LaunchArg>::register(shape_arg, launcher),
            physical_strides: <CoordsDyn as LaunchArg>::register(strides_arg, launcher),
            start_axis: tiler.start_axis,
            tile_shape: <CoordsDyn as LaunchArg>::register(tile_arg, launcher),
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        _ty: Type,
        builder: &mut KernelBuilder,
    ) -> <Self as CubeType>::ExpandType {
        TiledTensorLayoutExpand {
            physical_shape: <CoordsDyn as LaunchArg>::expand(&arg.physical_shape, builder),
            physical_strides: <CoordsDyn as LaunchArg>::expand(&arg.physical_strides, builder),
            start_axis: arg.start_axis as usize,
            tile_shape: <CoordsDyn as LaunchArg>::expand(&arg.tile_shape, builder),
        }
    }
}

/// Concrete launch-able version of [`TiledTensorLayout`].
pub type TiledTensorLayoutConcrete = ConcreteLayout<TiledTensorLayout>;
pub type TiledTensorLayoutLaunch<R> = ConcreteLayoutLaunch<TiledTensorLayout, R>;

/// View type alias for a tiled tensor's semantic coordinates.
pub type TiledTensorView<E, IO = cubecl_core::prelude::ReadOnly> = View<E, CoordsDyn, IO>;
pub type TiledTensorViewLaunch<R> = ViewArg<CoordsDyn, R>;

/// Build a launch arg for viewing a tiled tensor through semantic coordinates.
/// The handle must carry tiler metadata.
pub fn tiled_tensor_view<R: Runtime>(handle: TensorBinding<R>) -> TiledTensorViewLaunch<R> {
    TiledTensorViewLaunch::new_tensor::<TiledTensorLayout>(handle.into_tensor_arg(), ())
}
