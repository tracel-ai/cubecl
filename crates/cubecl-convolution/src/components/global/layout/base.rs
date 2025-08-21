macro_rules! virtual_layout {
    ($ty: ident, $expand: ident) => {
        mod r#virtual {
            use super::*;
            use cubecl_matmul::components::layout::*;
            type L<C> = $ty<C>;
            type Coords<C> = <L<C> as Layout>::Coordinates;
            type CoordsExpand<C> = <Coords<C> as CubeType>::ExpandType;

            impl<C: ConvGemmConfig> VirtualLayoutOperationsExpand<Coords<C>> for $expand<C> {
                fn __expand_to_linear_pos_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand<C>,
                ) -> <u32 as CubeType>::ExpandType {
                    L::<C>::__expand_to_linear_pos(scope, self.clone(), pos)
                }
                fn __expand_to_linear_pos_checked_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand<C>,
                ) -> <(u32, bool) as CubeType>::ExpandType {
                    L::<C>::__expand_to_linear_pos_checked(scope, self.clone(), pos)
                }
                fn __expand_shape_method(&self, scope: &mut Scope) -> CoordsExpand<C> {
                    L::<C>::__expand_shape(scope, self.clone())
                }
            }

            #[cube]
            impl<C: ConvGemmConfig> $ty<C> {
                pub fn into_virtual(self) -> VirtualLayout<Coords<C>> {
                    VirtualLayout::new::<L<C>>(self)
                }
            }
        }
    };
}

pub(crate) use virtual_layout;
