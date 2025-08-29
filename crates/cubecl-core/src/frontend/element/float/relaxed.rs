use cubecl_common::flex32;
use cubecl_ir::{ElemType, ExpandElement, FloatKind, Scope, StorageType};

use crate::prelude::{Numeric, into_runtime_expand_element};

use super::{
    CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, Float, IntoRuntime,
    KernelBuilder, KernelLauncher, LaunchArgExpand, Runtime, ScalarArgSettings,
    into_mut_expand_element,
};

impl CubeType for flex32 {
    type ExpandType = ExpandElementTyped<flex32>;
}

impl CubePrimitive for flex32 {
    /// Return the element type to use on GPU
    fn as_type_native() -> Option<StorageType> {
        Some(ElemType::Float(FloatKind::Flex32).into())
    }
}

impl IntoRuntime for flex32 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let elem: ExpandElementTyped<Self> = self.into();
        into_runtime_expand_element(scope, elem).into()
    }
}

impl Numeric for flex32 {
    fn min_value() -> Self {
        <Self as num_traits::Float>::min_value()
    }
    fn max_value() -> Self {
        <Self as num_traits::Float>::max_value()
    }
}

impl ExpandElementIntoMut for flex32 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        into_mut_expand_element(scope, elem)
    }
}

impl Float for flex32 {
    const DIGITS: u32 = 32;

    const EPSILON: Self = flex32::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = flex32::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = f32::MANTISSA_DIGITS;

    /// Maximum possible [`flex32`](crate::frontend::flex32) power of 10 exponent
    const MAX_10_EXP: i32 = f32::MAX_10_EXP;
    /// Maximum possible [`flex32`](crate::frontend::flex32) power of 2 exponent
    const MAX_EXP: i32 = f32::MAX_EXP;

    /// Minimum possible normal [`flex32`](crate::frontend::flex32) power of 10 exponent
    const MIN_10_EXP: i32 = f32::MIN_10_EXP;
    /// One greater than the minimum possible normal [`flex32`](crate::frontend::flex32) power of 2 exponent
    const MIN_EXP: i32 = f32::MIN_EXP;

    const MIN_POSITIVE: Self = flex32::from_f32(f32::MIN_POSITIVE);

    const NAN: Self = flex32::from_f32(f32::NAN);

    const NEG_INFINITY: Self = flex32::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        flex32::from_f32(val)
    }
}

impl LaunchArgExpand for flex32 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(flex32::as_type(&builder.scope)).into()
    }
}

impl ScalarArgSettings for flex32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32(self.to_f32());
    }
}
