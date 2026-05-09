use cubecl_core::{prelude::*, unexpanded};

use crate::tensor::{View, ViewExpand, layout::*};

type ArrayExpand<T> = NativeExpand<Array<T>>;
type SharedMemoryExpand<T> = NativeExpand<SharedMemory<T>>;

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
        scope: &Scope,
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
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly>;
}

pub trait AsViewMut<E: CubePrimitive>: AsView<E> {
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
        &mut self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadWrite>;
}

macro_rules! impl_as_view {
    ($ty: ident, $expand: ident, $coords: ty) => {
        impl<E: CubePrimitive> AsView<E> for $ty<E> {
            type SourceCoords = $coords;
        }
        impl<E: CubePrimitive> AsViewExpand<E> for $expand<E> {
            type SourceCoords = $coords;
            fn __expand_view_method<C: Coordinates + 'static>(
                &self,
                scope: &Scope,
                layout: VirtualLayoutExpand<C, $coords>,
            ) -> super::ViewExpand<E, C, ReadOnly> {
                View::__expand_new::<Box<[E]>, $coords>(
                    scope,
                    unsafe { self.__expand_as_boxed_unchecked_method(scope) },
                    layout,
                )
            }
        }

        impl<E: CubePrimitive> AsViewMut<E> for $ty<E> {}
        impl<E: CubePrimitive> AsViewMutExpand<E> for $expand<E> {
            fn __expand_view_mut_method<C: Coordinates + 'static>(
                &mut self,
                scope: &Scope,
                layout: VirtualLayoutExpand<C, $coords>,
            ) -> super::ViewExpand<E, C, ReadWrite> {
                View::__expand_new_mut::<Box<[E]>, $coords>(
                    scope,
                    unsafe {
                        self.__expand_as_mut_slice_method(scope)
                            .__expand_as_boxed_unchecked_method(scope)
                    },
                    layout,
                )
            }
        }
    };
}

impl_as_view!(Array, ArrayExpand, Coords1d);
impl_as_view!(Tensor, TensorExpand, Coords1d);
impl_as_view!(SharedMemory, SharedMemoryExpand, Coords1d);

impl<E: CubePrimitive> AsView<E> for [E] {
    type SourceCoords = Coords1d;
    fn view<C: Coordinates + 'static>(
        &self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<E, C, ReadOnly> {
        View::new::<Box<[E]>, Coords1d>(unsafe { self.as_boxed_unchecked() }, layout)
    }
}

impl<E: CubePrimitive> AsViewExpand<E> for SliceExpand<E> {
    type SourceCoords = Coords1d;
    fn __expand_view_method<C: Coordinates + 'static>(
        &self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadOnly> {
        View::__expand_new::<Box<[E]>, Self::SourceCoords>(
            scope,
            unsafe { self.__expand_as_boxed_unchecked_method(scope) },
            layout,
        )
    }
}

impl<E: CubePrimitive> AsViewMut<E> for [E] {
    fn view_mut<C: Coordinates + 'static>(
        &mut self,
        layout: impl Into<VirtualLayout<C, Coords1d>>,
    ) -> View<E, C, ReadWrite> {
        View::new_mut::<Box<[E]>, Coords1d>(unsafe { self.as_boxed_unchecked() }, layout)
    }
}
impl<E: CubePrimitive> AsViewMutExpand<E> for SliceExpand<E> {
    fn __expand_view_mut_method<C: Coordinates + 'static>(
        &mut self,
        scope: &Scope,
        layout: VirtualLayoutExpand<C, Self::SourceCoords>,
    ) -> ViewExpand<E, C, ReadWrite> {
        View::__expand_new_mut::<Box<[E]>, Coords1d>(
            scope,
            unsafe { self.__expand_as_boxed_unchecked_method(scope) },
            layout,
        )
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
                    ) -> View<E, C, ReadOnly> {
                        unexpanded!()
                    }

                    fn [<__expand_view_ $dim>]<C: Coordinates + 'static>(
                        scope: &Scope,
                        this: Self::ExpandType,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> ViewExpand<E, C, ReadOnly> {
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
                    ) -> ViewExpand<E, C, ReadOnly>;
                )*
            }

            pub trait AsTensorViewMut<E: CubePrimitive>: AsTensorView<E> {
                $(
                    #[allow(unused)]
                    fn [<view_mut_ $dim>]<C: Coordinates + 'static>(
                        &mut self,
                        layout: impl Into<VirtualLayout<C, [<Coords $dim>]>>,
                    ) -> View<E, C, ReadWrite> {
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
                    ) -> ViewExpand<E, C, ReadWrite>;
                )*
            }

            impl<E: CubePrimitive> AsTensorView<E> for TensorMap<E, Tiled> {}
            impl<E: CubePrimitive> AsTensorViewExpand<E> for NativeExpand<TensorMap<E, Tiled>> {
                $(
                    fn [<__expand_view_ $dim _method>]<C: Coordinates + 'static>(
                        self,
                        scope: &Scope,
                        layout: VirtualLayoutExpand<C, [<Coords $dim>]>,
                    ) -> super::ViewExpand<E, C, ReadOnly> {
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
                    ) -> super::ViewExpand<E, C, ReadWrite> {
                        View::__expand_new_mut::<TensorMap<E, Tiled>, [<Coords $dim>]>(scope, self, layout)
                    }
                )*
            }
        }
    };
}

as_view_tensor_map!(1d, 2d, 3d, 4d, 5d, 1i, 2i, 3i, 4i, 5i);
