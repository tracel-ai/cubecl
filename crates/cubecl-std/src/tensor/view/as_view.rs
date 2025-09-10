use cubecl_core::{
    prelude::{self, *},
    unexpanded,
};

use crate::tensor::{
    View, ViewExpand,
    layout::{Coordinates, Coords1d, VirtualLayout, VirtualLayoutExpand},
};

pub trait AsView<E: CubePrimitive> {
    #[allow(unused)]
    fn view<C: Coordinates, S: Coordinates>(
        &self,
        layout: VirtualLayout<C, S>,
    ) -> View<E, C, ReadOnly> {
        unexpanded!()
    }
}

pub trait AsViewExpand<E: CubePrimitive> {
    fn __expand_view_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<E, C, ReadOnly>;
}

pub trait AsViewMut<E: CubePrimitive> {
    #[allow(unused)]
    fn view_mut<C: Coordinates>(
        &mut self,
        layout: VirtualLayout<C, Coords1d>,
    ) -> View<E, C, ReadWrite> {
        unexpanded!()
    }
}

pub trait AsViewMutExpand<E: CubePrimitive> {
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> ViewExpand<E, C, ReadWrite>;
}

macro_rules! impl_as_view {
    ($ty: ident) => {
        impl<E: CubePrimitive> AsView<E> for $ty<E> {}
        impl<E: CubePrimitive> AsViewExpand<E> for ExpandElementTyped<$ty<E>> {
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

impl<E: CubePrimitive, IO: SliceVisibility> AsView<E> for Slice<E, IO> {}
impl<E: CubePrimitive, IO: SliceVisibility + 'static> AsViewExpand<E> for SliceExpand<E, IO> {
    fn __expand_view_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> super::ViewExpand<E, C, ReadOnly> {
        View::__expand_new::<Slice<E, IO>, Coords1d>(scope, self, layout)
    }
}

impl<E: CubePrimitive> AsViewMut<E> for Slice<E, prelude::ReadWrite> {}
impl<E: CubePrimitive> AsViewMutExpand<E> for SliceExpand<E, prelude::ReadWrite> {
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        self,
        scope: &mut Scope,
        layout: VirtualLayoutExpand<C, Coords1d>,
    ) -> super::ViewExpand<E, C, ReadWrite> {
        View::__expand_new_mut::<Slice<E, prelude::ReadWrite>, Coords1d>(scope, self, layout)
    }
}
