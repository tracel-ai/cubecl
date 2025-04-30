use cubecl_common::tf32;
use cubecl_ir::{Elem, ExpandElement, FloatKind, Scope};
use half::f16;

use crate::prelude::Numeric;

use super::{
    CubePrimitive, CubeType, ExpandElementIntoMut, ExpandElementTyped, Float, IntoMut, IntoRuntime,
    KernelBuilder, KernelLauncher, LaunchArgExpand, Runtime, ScalarArgSettings,
    into_mut_expand_element,
};

impl CubeType for tf32 {
    type ExpandType = ExpandElementTyped<tf32>;
}

impl CubePrimitive for tf32 {
    /// Return the element type to use on GPU
    fn as_elem_native() -> Option<Elem> {
        Some(Elem::Float(FloatKind::TF32))
    }
}

impl IntoRuntime for tf32 {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = self.into();
        IntoMut::into_mut(expand, scope, false)
    }
}

impl Numeric for tf32 {
    fn min_value() -> Self {
        Self::from_f32(f32::MIN)
    }
    fn max_value() -> Self {
        Self::from_f32(f32::MAX)
    }
}

impl ExpandElementIntoMut for tf32 {
    fn elem_into_mut(scope: &mut Scope, elem: ExpandElement, is_mut: bool) -> ExpandElement {
        into_mut_expand_element(scope, elem, is_mut)
    }
}

impl Float for tf32 {
    const DIGITS: u32 = 32;

    const EPSILON: Self = tf32::from_f32(half::f16::EPSILON.to_f32_const());

    const INFINITY: Self = tf32::from_f32(f32::INFINITY);

    const MANTISSA_DIGITS: u32 = 10;

    /// Maximum possible [`tf32`](crate::frontend::tf32) power of 10 exponent
    const MAX_10_EXP: i32 = 38;
    /// Maximum possible [`tf32`](crate::frontend::tf32) power of 2 exponent
    const MAX_EXP: i32 = 128;

    /// Minimum possible normal [`tf32`](crate::frontend::tf32) power of 10 exponent
    const MIN_10_EXP: i32 = -37;
    /// One greater than the minimum possible normal [`tf32`](crate::frontend::tf32) power of 2 exponent
    const MIN_EXP: i32 = -125;

    /// `MIN_POSITIVE` is defined by precision, so use `f16` as reference
    const MIN_POSITIVE: Self = tf32::from_f32(f16::MIN_POSITIVE.to_f32_const());

    const NAN: Self = tf32::from_f32(f32::NAN);

    const NEG_INFINITY: Self = tf32::from_f32(f32::NEG_INFINITY);

    const RADIX: u32 = 2;

    fn new(val: f32) -> Self {
        tf32::from_f32(val)
    }
}

impl ScalarArgSettings for tf32 {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_f32((*self).to_f32());
    }
}

impl LaunchArgExpand for tf32 {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder.scalar(tf32::as_elem(&builder.scope)).into()
    }
}
