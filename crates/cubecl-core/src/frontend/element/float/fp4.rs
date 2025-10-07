use cubecl_common::{e2m1, e2m1x2};
use cubecl_ir::{ElemType, ExpandElement, FloatKind, Scope, StorageType};

use crate::{
    Runtime,
    compute::KernelLauncher,
    prelude::{
        CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, IntoRuntime,
        ScalarArgSettings, into_mut_expand_element, into_runtime_expand_element,
    },
};

impl CubeType for e2m1 {
    type ExpandType = ExpandElementTyped<e2m1>;
}

impl CubePrimitive for e2m1 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<StorageType> {
        Some(StorageType::Scalar(ElemType::Float(FloatKind::E2M1)))
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
    fn as_type_native() -> Option<StorageType> {
        Some(StorageType::Packed(ElemType::Float(FloatKind::E2M1), 2))
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
