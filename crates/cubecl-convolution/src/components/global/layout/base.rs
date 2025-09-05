macro_rules! virtual_layout {
    ($ty: ident, $expand: ident) => {
        mod r#virtual {
            use super::*;
            use cubecl_std::tensor::layout::*;
            type L = $ty;
            type Coords = <L as Layout>::Coordinates;
            type SourceCoords = <L as Layout>::SourceCoordinates;
            type CoordsExpand = <Coords as CubeType>::ExpandType;

            impl VirtualLayoutOperationsExpand<Coords, SourceCoords> for $expand {
                fn __expand_to_source_pos_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <SourceCoords as CubeType>::ExpandType {
                    L::__expand_to_source_pos(scope, self.clone(), pos)
                }
                fn __expand_to_source_pos_checked_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <(SourceCoords, bool) as CubeType>::ExpandType {
                    L::__expand_to_source_pos_checked(scope, self.clone(), pos)
                }
                fn __expand_shape_method(&self, scope: &mut Scope) -> CoordsExpand {
                    L::__expand_shape(scope, self.clone())
                }
                fn __expand_is_in_bounds_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> ExpandElementTyped<bool> {
                    L::__expand_is_in_bounds(scope, self.clone(), pos)
                }
            }

            #[cube]
            impl $ty {
                pub fn virt(self) -> VirtualLayout<Coords, SourceCoords> {
                    VirtualLayout::new::<L>(self)
                }
            }
        }
    };
}

pub(crate) use virtual_layout;
