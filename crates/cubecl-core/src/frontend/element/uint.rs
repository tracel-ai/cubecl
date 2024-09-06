use crate::frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement, Numeric};
use crate::ir::{Elem, Vectorization};
use crate::prelude::{KernelBuilder, KernelLauncher};
use crate::Runtime;

use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, LaunchArgExpand,
    ScalarArgSettings,
};

impl CubeType for u32 {
    type ExpandType = ExpandElementTyped<Self>;
}

impl ExpandElementBaseInit for u32 {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl CubePrimitive for u32 {
    fn as_elem() -> Elem {
        Elem::UInt
    }
}

impl LaunchArgExpand for u32 {
    fn expand(
        builder: &mut KernelBuilder,
        vectorization: Vectorization,
    ) -> ExpandElementTyped<Self> {
        assert_eq!(vectorization, None, "Attempted to vectorize a scalar");
        builder.scalar(u32::as_elem()).into()
    }
}

impl ScalarArgSettings for u32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u32(*self);
    }
}

impl Numeric for u32 {}
