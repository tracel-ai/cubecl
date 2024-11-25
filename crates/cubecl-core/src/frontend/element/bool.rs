use crate::compute::{KernelBuilder, KernelLauncher};
use crate::frontend::{CubePrimitive, CubeType};
use crate::ir::Elem;
use crate::prelude::CubeContext;
use crate::{unexpanded, Runtime};

use super::{
    init_expand_element, ExpandElement, ExpandElementBaseInit, ExpandElementTyped, Init,
    IntoRuntime, LaunchArgExpand, Numeric, ScalarArgSettings, Vectorized,
};

/// Extension trait for [bool].
pub trait BoolOps {
    #[allow(clippy::new_ret_no_self)]
    fn new(value: bool) -> bool {
        value
    }
    fn __expand_new(
        _context: &mut CubeContext,
        value: ExpandElementTyped<bool>,
    ) -> ExpandElementTyped<bool> {
        ExpandElement::Plain(Elem::Bool.from_constant(*value.expand)).into()
    }
}

impl BoolOps for bool {}

impl CubeType for bool {
    type ExpandType = ExpandElementTyped<Self>;
}

impl CubePrimitive for bool {
    fn as_elem() -> Elem {
        Elem::Bool
    }
}

impl IntoRuntime for bool {
    fn __expand_runtime_method(self, context: &mut CubeContext) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        Init::init(expand, context)
    }
}

impl ExpandElementBaseInit for bool {
    fn init_elem(context: &mut CubeContext, elem: ExpandElement) -> ExpandElement {
        init_expand_element(context, elem)
    }
}

impl Vectorized for bool {
    fn vectorization_factor(&self) -> u32 {
        1
    }

    fn vectorize(self, _factor: u32) -> Self {
        unexpanded!()
    }
}

impl LaunchArgExpand for bool {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(bool::as_elem()).into()
    }
}

impl Numeric for bool {
    const MAX: Self = true;
    const MIN: Self = false;

    fn from_int(val: i64) -> Self {
        val != 0
    }
}

impl ScalarArgSettings for bool {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_u32(if *self { 1 } else { 0 });
    }
}
