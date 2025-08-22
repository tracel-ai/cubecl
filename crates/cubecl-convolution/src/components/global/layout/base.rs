macro_rules! virtual_layout {
    ($ty: ident, $expand: ident) => {
        mod r#virtual {
            use super::*;
            use cubecl_std::tensor::layout::*;
            type L = $ty;
            type Coords = <L as Layout>::Coordinates;
            type CoordsExpand = <Coords as CubeType>::ExpandType;

            impl VirtualLayoutOperationsExpand<Coords> for $expand {
                fn __expand_to_linear_pos_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <u32 as CubeType>::ExpandType {
                    L::__expand_to_linear_pos(scope, self.clone(), pos)
                }
                fn __expand_to_linear_pos_checked_method(
                    &self,
                    scope: &mut Scope,
                    pos: CoordsExpand,
                ) -> <(u32, bool) as CubeType>::ExpandType {
                    L::__expand_to_linear_pos_checked(scope, self.clone(), pos)
                }
                fn __expand_shape_method(&self, scope: &mut Scope) -> CoordsExpand {
                    L::__expand_shape(scope, self.clone())
                }
            }

            #[cube]
            impl $ty {
                pub fn virt(self) -> VirtualLayout<Coords> {
                    VirtualLayout::new::<L>(self)
                }
            }
        }
    };
}

pub(crate) use virtual_layout;
