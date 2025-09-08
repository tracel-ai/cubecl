use cubecl_core as cubecl;
use cubecl_core::{prelude::*, unexpanded};

use crate::tensor::{
    View, ViewExpand,
    layout::{Coordinates, Coords1d, VirtualLayout, VirtualLayoutExpand},
};

pub trait AsView<E: CubePrimitive>:
    CubeType<ExpandType: AsViewExpand<E, SourceCoords = Self::SourceCoords>>
{
    type SourceCoords: Coordinates;

    #[allow(unused)]
    fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Self::SourceCoords>>,
    ) -> View<E, C, ReadOnly> {
        unexpanded!()
    }

    fn __expand_view<C: Coordinates + 'static>(
        scope: &mut Scope,
        this: Self::ExpandType,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly> {
        this.__expand_view_method(scope, layout)
    }
}

pub trait AsViewExpand<E: CubePrimitive> {
    type SourceCoords: Coordinates;

    #[allow(unused)]
    fn __expand_view_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly>;
}

pub trait AsViewMut<E: CubePrimitive>:
    AsView<E> + CubeType<ExpandType: AsViewExpand<E> + AsViewMutExpand<E>>
{
    #[allow(unused)]
    fn view_mut<C: Coordinates + 'static>(
        &mut self,
        layout: impl Into<VirtualLayout<C, Self::SourceCoords>>,
    ) -> View<E, C, ReadWrite> {
        unexpanded!()
    }
}

pub trait AsViewMutExpand<E: CubePrimitive>: AsViewExpand<E> {
    #[allow(clippy::too_many_arguments)]
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadWrite>;
}

macro_rules! impl_as_view {
    ($ty: ident) => {
        impl<E: CubePrimitive> AsView<E> for $ty<E> {
            type SourceCoords = Coords1d;
        }
        impl<E: CubePrimitive> AsViewExpand<E> for ExpandElementTyped<$ty<E>> {
            type SourceCoords = Coords1d;
            fn __expand_view_method<C: Coordinates + 'static>(
                self,
                scope: &mut Scope,
                layout: VirtualLayoutExpand<C, Coords1d>,
            ) -> super::ViewExpand<E, C, ReadOnly> {
                View::__expand_new::<$ty<E>, Coords1d>(scope, self, layout)
            }
        }

        impl<E: CubePrimitive> AsViewMut<E> for $ty<E> {}
        impl<E: CubePrimitive> AsViewMutExpand<E> for ExpandElementTyped<$ty<E>> {
            fn __expand_view_mut_method<C: Coordinates + 'static>(
                self,
                scope: &mut Scope,
                layout: VirtualLayoutExpand<C, Coords1d>,
            ) -> super::ViewExpand<E, C, ReadWrite> {
                View::__expand_new_mut::<$ty<E>, Coords1d>(scope, self, layout)
            }
        }
    };
}

impl_as_view!(Array);
impl_as_view!(Tensor);
impl_as_view!(SharedMemory);

impl<E: CubePrimitive, IO: SliceVisibility + 'static> AsView<E> for Slice<E, IO> {
    type SourceCoords = Coords1d;
    fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<E, C, ReadOnly> {
        View::new::<Slice<E, IO>, Coords1d>(self, layout)
    }
}

impl<E: CubePrimitive, IO: SliceVisibility + 'static> AsViewExpand<E> for SliceExpand<E, IO> {
    type SourceCoords = Coords1d;
    fn __expand_view_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly> {
        View::__expand_new::<Slice<E, IO>, Self::SourceCoords>(scope, self, layout)
    }
}

impl<E: CubePrimitive> AsViewMut<E> for Slice<E, ReadWrite> {
    fn view_mut<C: Coordinates + 'static>(
        &mut self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<E, C, ReadWrite> {
        View::new_mut::<Slice<E, ReadWrite>, Coords1d>(self, layout)
    }
}
impl<E: CubePrimitive> AsViewMutExpand<E> for SliceExpand<E, ReadWrite> {
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &mut cubecl::prelude::Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadWrite> {
        View::__expand_new_mut::<Slice<E, ReadWrite>, Coords1d>(scope, self, layout)
    }
}
