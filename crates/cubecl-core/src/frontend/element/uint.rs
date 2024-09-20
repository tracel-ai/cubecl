use crate::frontend::{CubeContext, CubePrimitive, CubeType, ExpandElement, Numeric};
use crate::ir::Elem;
use crate::prelude::{KernelBuilder, KernelLauncher};
use crate::Runtime;

use super::{
    init_expand_element, ExpandElementBaseInit, ExpandElementTyped, Init, IntoRuntime,
    LaunchArgExpand, ScalarArgSettings,
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

impl IntoRuntime for u32 {
    fn __expand_runtime_method(self, context: &mut CubeContext) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        Init::init(expand, context)
    }
}

impl LaunchArgExpand for u32 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(u32::as_elem()).into()
    }
}

impl ScalarArgSettings for u32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u32(*self);
    }
}

impl Numeric for u32 {
    const MAX: Self = u32::MAX;
    const MIN: Self = u32::MIN;
}
