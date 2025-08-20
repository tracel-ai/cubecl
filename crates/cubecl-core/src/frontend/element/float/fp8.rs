use cubecl_common::{e4m3, e5m2, ue8m0};
use cubecl_ir::{Elem, ExpandElement, FloatKind, Scope};

use crate::{
    Runtime,
    compute::{KernelBuilder, KernelLauncher},
    prelude::{
        CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, IntoRuntime,
        LaunchArgExpand, Numeric, ScalarArgSettings, into_mut_expand_element,
        into_runtime_expand_element,
    },
};

impl CubeType for e4m3 {
    type ExpandType = ExpandElementTyped<e4m3>;
}

impl CubePrimitive for e4m3 {
    /// Return the element type to use on GPU
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Float(FloatKind::E4M3))
    }
}

impl IntoRuntime for e4m3 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl Numeric for e4m3 {
    fn min_value() -> Self {
        Self::from_f64(Self::MIN)
    }
    fn max_value() -> Self {
        Self::from_f64(Self::MAX)
    }
}

impl ExpandElementIntoMut for e4m3 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl ScalarArgSettings for e4m3 {
    fn register<R: Runtime>(&self, _settings: &mut KernelLauncher<R>) {
        todo!("Not yet supported for scalars")
    }
}

impl LaunchArgExpand for e4m3 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, _builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        todo!("Not yet supported for scalars")
    }
}

impl CubeType for e5m2 {
    type ExpandType = ExpandElementTyped<e5m2>;
}

impl CubePrimitive for e5m2 {
    /// Return the element type to use on GPU
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Float(FloatKind::E5M2))
    }
}

impl IntoRuntime for e5m2 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl Numeric for e5m2 {
    fn min_value() -> Self {
        Self::from_f64(Self::MIN)
    }
    fn max_value() -> Self {
        Self::from_f64(Self::MAX)
    }
}

impl ExpandElementIntoMut for e5m2 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl ScalarArgSettings for e5m2 {
    fn register<R: Runtime>(&self, _settings: &mut KernelLauncher<R>) {
        todo!("Not yet supported for scalars")
    }
}

impl LaunchArgExpand for e5m2 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, _builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        todo!("Not yet supported for scalars")
    }
}

impl CubeType for ue8m0 {
    type ExpandType = ExpandElementTyped<ue8m0>;
}

impl CubePrimitive for ue8m0 {
    /// Return the element type to use on GPU
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Float(FloatKind::UE8M0))
    }
}

impl IntoRuntime for ue8m0 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl Numeric for ue8m0 {
    fn min_value() -> Self {
        Self::from_f64(Self::MIN)
    }
    fn max_value() -> Self {
        Self::from_f64(Self::MAX)
    }
}

impl ExpandElementIntoMut for ue8m0 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl ScalarArgSettings for ue8m0 {
    fn register<R: Runtime>(&self, _settings: &mut KernelLauncher<R>) {
        todo!("Not yet supported for scalars")
    }
}

impl LaunchArgExpand for ue8m0 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, _builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        todo!("Not yet supported for scalars")
    }
}
