use cubecl::prelude::*;
use cubecl_core as cubecl;

use crate::tensor::layout::{Layout, LayoutExpand};

/// Chain of layouts, can be used to launch with multiple layouts
#[derive(CubeType)]
pub struct Chain<L0: Layout, L1: Layout<SourceCoordinates = L0::Coordinates>> {
    l0: L0,
    l1: L1,
}

#[cube]
impl<L0: Layout, L1: Layout<SourceCoordinates = L0::Coordinates>> Chain<L0, L1> {
    pub fn new(l0: L0, l1: L1) -> Self {
        Chain::<L0, L1> { l0, l1 }
    }
}

#[cube]
impl<L0: Layout, L1: Layout<SourceCoordinates = L0::Coordinates>> Layout for Chain<L0, L1> {
    type Coordinates = L1::Coordinates;
    type SourceCoordinates = L0::SourceCoordinates;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let pos = self.l1.to_source_pos(pos);
        self.l0.to_source_pos(pos)
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (pos, l1_in_bounds) = self.l1.to_source_pos_checked(pos);
        self.l0.is_in_bounds(pos) && l1_in_bounds
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let (pos, l1_in_bounds) = self.l1.to_source_pos_checked(pos);
        let (pos, l0_in_bounds) = self.l0.to_source_pos_checked(pos);
        (pos, l0_in_bounds && l1_in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.l1.shape()
    }
}

pub use launch::*;
mod launch {
    use core::marker::PhantomData;

    use crate::tensor::launch::{BufferArg, ViewLayoutLaunchArg};

    use super::*;

    pub struct ChainLaunch<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
        R: Runtime,
    > {
        _phantom_runtime: PhantomData<R>,
        l0: L0::RuntimeArg<R>,
        l1: L1::RuntimeArg<R>,
    }
    impl<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
        R: Runtime,
    > ChainLaunch<L0, L1, R>
    {
        pub fn new(l0: L0::RuntimeArg<R>, l1: L1::RuntimeArg<R>) -> Self {
            Self {
                _phantom_runtime: PhantomData,
                l0,
                l1,
            }
        }
    }

    pub struct ChainCompilationArg<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
    > {
        l0: L0::CompilationArg,
        l1: L1::CompilationArg,
    }
    impl<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
    > Clone for ChainCompilationArg<L0, L1>
    {
        fn clone(&self) -> Self {
            Self {
                l0: self.l0.clone(),
                l1: self.l1.clone(),
            }
        }
    }

    impl<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
    > core::hash::Hash for ChainCompilationArg<L0, L1>
    {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.l0.hash(state);
            self.l1.hash(state);
        }
    }
    impl<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
    > core::cmp::PartialEq for ChainCompilationArg<L0, L1>
    {
        fn eq(&self, other: &Self) -> bool {
            self.l0.eq(&other.l0) && self.l1.eq(&other.l1)
        }
    }
    impl<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
    > core::fmt::Debug for ChainCompilationArg<L0, L1>
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct(stringify!(Chain))
                .field(stringify!(l0), &self.l0)
                .field(stringify!(l1), &self.l1)
                .finish()
        }
    }
    impl<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
    > core::cmp::Eq for ChainCompilationArg<L0, L1>
    {
    }

    impl<
        L0: Layout + ViewLayoutLaunchArg,
        L1: Layout<SourceCoordinates = L0::Coordinates> + ViewLayoutLaunchArg,
    > ViewLayoutLaunchArg for Chain<L0, L1>
    {
        type RuntimeArg<R: Runtime> = ChainLaunch<L0, L1, R>;
        type CompilationArg = ChainCompilationArg<L0, L1>;

        fn register<R: Runtime, B: BufferArg>(
            arg: Self::RuntimeArg<R>,
            buffer: &B,
            ty: Type,
            launcher: &mut KernelLauncher<R>,
        ) -> Self::CompilationArg {
            ChainCompilationArg {
                l0: L0::register(arg.l0, buffer, ty, launcher),
                l1: L1::register(arg.l1, buffer, ty, launcher),
            }
        }
        fn expand(
            arg: &Self::CompilationArg,
            ty: Type,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            ChainExpand {
                l0: L0::expand(&arg.l0, ty, builder),
                l1: L1::expand(&arg.l1, ty, builder),
            }
        }
        fn expand_output(
            arg: &Self::CompilationArg,
            ty: Type,
            builder: &mut KernelBuilder,
        ) -> <Self as CubeType>::ExpandType {
            ChainExpand {
                l0: L0::expand_output(&arg.l0, ty, builder),
                l1: L1::expand_output(&arg.l1, ty, builder),
            }
        }
    }
}
