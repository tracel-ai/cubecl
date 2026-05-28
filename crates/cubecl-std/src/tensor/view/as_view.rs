use cubecl_core::{prelude::*, unexpanded};

use crate::tensor::{View, ViewExpand, ViewMut, ViewMutExpand, layout::*};

type ArrayExpand<T> = NativeExpand<Array<T>>;

pub trait AsView<E: CubePrimitive>:
    CubeType<ExpandType: AsViewExpand<E, SourceCoords = Self::SourceCoords>>
{
    type SourceCoords: Coordinates;

    #[allow(unused)]
    fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Self::SourceCoords>>,
    ) -> View<'_, E, C> {
        unexpanded!()
    }

    fn __expand_view<'a, C: Coordinates + 'static>(
        scope: &Scope,
        this: &'a Self::ExpandType,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<'a, E, C> {
        this.__expand_view_method(scope, layout)
    }
}

pub trait AsViewExpand<E: CubePrimitive> {
    type SourceCoords: Coordinates;

    #[allow(unused)]
    fn __expand_view_method<C: Coordinates + 'static>(
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<'_, E, C>;
}

pub trait AsViewMut<E: CubePrimitive>: AsView<E> {
    #[allow(unused)]
    fn view_mut<C: Coordinates + 'static>(
        &mut self,
        layout: impl Into<VirtualLayout<C, Self::SourceCoords>>,
    ) -> ViewMut<'_, E, C> {
        unexpanded!()
    }
}

pub trait AsViewMutExpand<E: CubePrimitive>: AsViewExpand<E> {
    #[allow(clippy::too_many_arguments)]
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        &mut self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewMutExpand<'_, E, C>;
}

macro_rules! impl_as_view {
    ($ty: ty, $expand: ty, $coords: ty) => {
        impl<E: CubePrimitive> AsView<E> for $ty {
            type SourceCoords = $coords;
        }
        impl<E: CubePrimitive> AsViewExpand<E> for $expand {
            type SourceCoords = $coords;
            fn __expand_view_method<C: Coordinates + 'static>(
                &self,
                scope: &Scope,
                layout: VirtualLayoutExpand<C, $coords>,
            ) -> super::ViewExpand<'_, E, C> {
                View::__expand_new::<&[E], $coords>(
                    scope,
                    self.__expand_as_slice_method(scope),
                    layout,
                )
            }
        }

        impl<E: CubePrimitive> AsViewMut<E> for $ty {}
        impl<E: CubePrimitive> AsViewMutExpand<E> for $expand {
            fn __expand_view_mut_method<C: Coordinates + 'static>(
                &mut self,
                scope: &Scope,
                layout: VirtualLayoutExpand<C, $coords>,
            ) -> super::ViewMutExpand<'_, E, C> {
                ViewMut::__expand_new::<&mut [E], $coords>(
                    scope,
                    self.__expand_as_mut_slice_method(scope),
                    layout,
                )
            }
        }
    };
}

impl_as_view!(Array<E>, ArrayExpand<E>, Coords1d);
impl_as_view!(Tensor<E>, TensorExpand<E>, Coords1d);
impl_as_view!(Shared<[E]>, SharedExpand<[E]>, Coords1d);

impl<E: CubePrimitive> AsView<E> for [E] {
    type SourceCoords = Coords1d;
    fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<'_, E, C> {
        View::new::<&[E], Coords1d>(self, layout)
    }
}

impl<E: CubePrimitive> AsViewExpand<E> for SliceExpand<E> {
    type SourceCoords = Coords1d;
    fn __expand_view_method<C: Coordinates + 'static>(
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<'_, E, C> {
        View::__expand_new::<&[E], Self::SourceCoords>(scope, self, layout)
    }
}

impl<E: CubePrimitive> AsViewMut<E> for [E] {
    fn view_mut<C: Coordinates + 'static>(
        &mut self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> ViewMut<'_, E, C> {
        ViewMut::new::<&mut [E], Coords1d>(self, layout)
    }
}
impl<E: CubePrimitive> AsViewMutExpand<E> for SliceExpand<E> {
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        &mut self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewMutExpand<'_, E, C> {
        ViewMut::__expand_new::<&mut [E], Coords1d>(scope, self, layout)
    }
}

macro_rules! as_view_tensor_map {
    ($($dim: literal),*) => {
        paste::paste! {
            pub trait AsTensorView<E: CubePrimitive>:
                CubeType<ExpandType: AsTensorViewExpand<E>>
            {
                $(
                    #[allow(unused)]
                    fn [<view_ $dim>]<C: Coordinates + 'static>(
                        &self,
                        layout: impl Into<VirtualLayout<C, [<Coords $dim>]>>,
                    ) -> View<'_, E, C> {
                        unexpanded!()
                    }

                    fn [<__expand_view_ $dim>]<C: Coordinates + 'static>(
                        scope: &Scope,
                        this: Self::ExpandType,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> ViewExpand<'_, E, C> {
                        this.[<__expand_view_ $dim _method>](scope, layout)
                    }
                )*
            }

            pub trait AsTensorViewExpand<E: CubePrimitive> {
                $(
                    #[allow(unused)]
                    fn [<__expand_view_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> ViewExpand<'_, E, C>;
                )*
            }

            pub trait AsTensorViewMut<E: CubePrimitive>: AsTensorView<E> {
                $(
                    #[allow(unused)]
                    fn [<view_mut_ $dim>]<C: Coordinates + 'static>(
                        &mut self,
                        layout: impl Into<VirtualLayout<C, [<Coords $dim>]>>,
                    ) -> ViewMut<'_, E, C> {
                        unexpanded!()
                    }
                )*
            }

            pub trait AsTensorViewMutExpand<E: CubePrimitive>: AsTensorViewExpand<E> {
                $(
                    #[allow(clippy::too_many_arguments)]
                    fn [<__expand_view_mut_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> ViewMutExpand<'_, E, C>;
                )*
            }

            impl<E: CubePrimitive> AsTensorView<E> for TensorMap<E, Tiled> {}
            impl<E: CubePrimitive> AsTensorViewExpand<E> for NativeExpand<TensorMap<E, Tiled>> {
                $(
                    fn [<__expand_view_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> super::ViewExpand<'_, E, C> {
                        View::__expand_new::<TensorMap<E, Tiled>, [<Coords $dim>]>(scope, self, layout)
                    }
                )*
            }

            impl<E: CubePrimitive> AsTensorViewMut<E> for TensorMap<E, Tiled> {}
            impl<E: CubePrimitive> AsTensorViewMutExpand<E> for NativeExpand<TensorMap<E, Tiled>> {
                $(
                    fn [<__expand_view_mut_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> super::ViewMutExpand<'_, E, C> {
                        ViewMut::__expand_new::<TensorMap<E, Tiled>, [<Coords $dim>]>(scope, self, layout)
                    }
                )*
            }
        }
    };
}

as_view_tensor_map!(1d, 2d, 3d, 4d, 5d, 1i, 2i, 3i, 4i, 5i);
