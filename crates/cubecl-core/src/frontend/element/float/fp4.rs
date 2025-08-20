use cubecl_common::{e2m1, e2m1x2};
use cubecl_ir::{Elem, ExpandElement, FloatKind, Scope};

use crate::{
    Runtime,
    compute::{KernelBuilder, KernelLauncher},
    prelude::{
        CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, IntoRuntime,
        LaunchArgExpand, ScalarArgSettings, into_mut_expand_element, into_runtime_expand_element,
    },
};

impl CubeType for e2m1 {
    type ExpandType = ExpandElementTyped<e2m1>;
}

impl CubePrimitive for e2m1 {
    /// Return the element type to use on GPU
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Float(FloatKind::E2M1))
    }
}

impl IntoRuntime for e2m1 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementIntoMut for e2m1 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl CubeType for e2m1x2 {
    type ExpandType = ExpandElementTyped<e2m1x2>;
}

impl CubePrimitive for e2m1x2 {
    /// Return the element type to use on GPU
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Float(FloatKind::E2M1x2))
    }
}

impl IntoRuntime for e2m1x2 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl ExpandElementIntoMut for e2m1x2 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl ScalarArgSettings for e2m1x2 {
    fn register<R: Runtime>(&self, _settings: &mut KernelLauncher<R>) {
        todo!("Not yet supported for scalars")
    }
}

impl LaunchArgExpand for e2m1x2 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, _builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        todo!("Not yet supported for scalars")
    }
}
