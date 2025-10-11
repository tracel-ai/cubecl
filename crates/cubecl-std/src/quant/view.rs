use std::marker::PhantomData;

use super::*;
use crate::{
    CubeOption, CubeOptionExpand,
    tensor::{
        View, ViewExpand, ViewOperations, ViewOperationsExpand, launch::ViewCompilationArg,
        layout::Coordinates,
    },
};
use cubecl::prelude::*;
use cubecl_common::{
    e2m1x2, e4m3, e5m2,
    quant::scheme::{QuantParam, QuantScheme, QuantStore, QuantValue},
    ue8m0,
};
use cubecl_core::{
    self as cubecl,
    ir::{ElemType, FloatKind, StorageType},
    prelude::barrier::BarrierExpand,
    unexpanded,
};
use half::{bf16, f16};

/// View that dequantizes after loads. Scales layout should take values coordinates and map them
/// to the corresponding scale.
/// Assumes only one scale maps to a single load. Adjust line size of values or block size to ensure
/// this.
#[expect(dead_code, reason = "only used in expand")]
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct QuantizedView<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static> {
    values: View<Line<Q>, C>,
    scales: View<S, C>,
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    _ty: PhantomData<F>,
}

#[cube]
impl<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static>
    QuantizedView<Q, S, F, C>
{
    pub fn new(
        values: View<Line<Q>, C>,
        scales: View<S, C>,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<Q, S, F, C> {
            values,
            scales,
            scheme,
            _ty: PhantomData,
        }
    }
}

impl<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static>
    QuantizedView<Q, S, F, C>
{
    pub fn view(self) -> View<Line<F>, C> {
        unexpanded!()
    }

    pub fn __expand_view(
        scope: &mut Scope,
        this: QuantizedViewExpand<Q, S, F, C>,
    ) -> ViewExpand<Line<F>, C, ReadOnly> {
        this.__expand_view_method(scope)
    }
}

impl<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static>
    QuantizedViewExpand<Q, S, F, C>
{
    pub fn new(
        values: ViewExpand<Line<Q>, C>,
        scales: ViewExpand<S, C>,
        scheme: QuantScheme,
    ) -> Self {
        QuantizedViewExpand::<Q, S, F, C> {
            values,
            scales,
            scheme,
            _ty: PhantomData,
        }
    }

    pub fn __expand_view_method(self, _scope: &mut Scope) -> ViewExpand<Line<F>, C, ReadOnly> {
        ViewExpand::new(self)
    }
}

impl<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static> Lined
    for QuantizedView<Q, S, F, C>
{
}
impl<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static> LinedExpand
    for QuantizedViewExpand<Q, S, F, C>
{
    fn line_size(&self) -> u32 {
        self.values.line_size() * self.scheme.num_quants() as u32
    }
}

impl<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static>
    ViewOperations<Line<F>, C> for QuantizedView<Q, S, F, C>
{
}

impl<Q: CubePrimitive, S: CubePrimitive, F: Float, C: Coordinates + 'static>
    ViewOperationsExpand<Line<F>, C> for QuantizedViewExpand<Q, S, F, C>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> ExpandElementTyped<Line<F>> {
        let value = self.values.clone().__expand_read_method(scope, pos.clone());
        let scale = self.scales.clone().__expand_read_method(scope, pos);

        dequantize_aligned::expand::<Q, S, F>(scope, value, scale, self.scheme)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> ExpandElementTyped<Line<F>> {
        let value = self
            .values
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let scale = self
            .scales
            .clone()
            .__expand_read_checked_method(scope, pos.clone());

        dequantize_aligned::expand::<Q, S, F>(scope, value, scale, self.scheme)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        mask_value: ExpandElementTyped<Line<F>>,
    ) -> ExpandElementTyped<Line<F>> {
        let value = self
            .values
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let scale = self
            .scales
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);

        let value = dequantize_aligned::expand::<Q, S, F>(scope, value, scale, self.scheme);
        select::expand::<Line<F>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> ExpandElementTyped<Line<F>> {
        let value = self
            .values
            .clone()
            .__expand_read_unchecked_method(scope, pos.clone());
        let scale = self
            .scales
            .clone()
            .__expand_read_unchecked_method(scope, pos);

        dequantize_aligned::expand::<Q, S, F>(scope, value, scale, self.scheme)
    }

    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _pos: <C>::ExpandType,
        _end: <C>::ExpandType,
    ) -> SliceExpand<Line<F>, ReadOnly> {
        panic!("Can't create raw slice for quantized view")
    }

    fn __expand_as_tensor_map_method(
        &self,
        scope: &mut Scope,
    ) -> CubeOptionExpand<TensorMap<Line<F>>> {
        CubeOption::__expand_new_None(scope)
    }

    fn __expand_shape_method(&self, scope: &mut Scope) -> <C>::ExpandType {
        self.values.clone().__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: C::ExpandType,
    ) -> ExpandElementTyped<bool> {
        self.values.clone().__expand_is_in_bounds_method(scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: BarrierExpand,
        _shared_memory: SliceExpand<Line<F>, ReadWrite>,
        _pos: C::ExpandType,
    ) {
        panic!("Can't use tensor map functions on quantized view");
    }
}

/// Dynamically expand based on the quantization scheme. Ugly, but the only way to fully hide the
/// quantization from the kernel using the view.
pub(crate) fn expand_dynamic<E: CubePrimitive, C: Coordinates + 'static, IO: SliceVisibility>(
    values: &ViewCompilationArg<C>,
    scales: &ViewCompilationArg<C>,
    scheme: QuantScheme,
    builder: &mut KernelBuilder,
) -> ViewExpand<E, C, IO> {
    use core::mem::transmute as t;

    // To specify tighter trait bounds
    fn expand_dynamic_f<F: Float, C: Coordinates + 'static>(
        values: &ViewCompilationArg<C>,
        scales: &ViewCompilationArg<C>,
        scheme: QuantScheme,
        builder: &mut KernelBuilder,
    ) -> ViewExpand<Line<F>, C> {
        let expand_q = match scheme.store {
            QuantStore::Native => match scheme.value {
                QuantValue::Q8F => expand_dynamic_q::<F, i8, C>,
                QuantValue::Q8S => expand_dynamic_q::<F, i8, C>,
                QuantValue::E5M2 => expand_dynamic_q::<F, e5m2, C>,
                QuantValue::E4M3 => expand_dynamic_q::<F, e4m3, C>,
                QuantValue::E2M1 => expand_dynamic_q::<F, e2m1x2, C>,
                QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S => {
                    panic!("Sub-byte quantization can't be native")
                }
            },
            QuantStore::U32 => expand_dynamic_q::<F, u32, C>,
        };
        expand_q(values, scales, scheme, builder)
    }

    fn expand_dynamic_q<F: Float, Q: CubePrimitive, C: Coordinates + 'static>(
        values: &ViewCompilationArg<C>,
        scales: &ViewCompilationArg<C>,
        scheme: QuantScheme,
        builder: &mut KernelBuilder,
    ) -> ViewExpand<Line<F>, C> {
        let expand_s = match scheme.param {
            QuantParam::F32 => expand_dynamic_s::<F, Q, f32, C>,
            QuantParam::F16 => expand_dynamic_s::<F, Q, f16, C>,
            QuantParam::BF16 => expand_dynamic_s::<F, Q, bf16, C>,
            QuantParam::UE8M0 => expand_dynamic_s::<F, Q, ue8m0, C>,
            QuantParam::UE4M3 => expand_dynamic_s::<F, Q, e4m3, C>,
        };
        expand_s(values, scales, scheme, builder)
    }

    fn expand_dynamic_s<F: Float, Q: CubePrimitive, S: CubePrimitive, C: Coordinates + 'static>(
        values: &ViewCompilationArg<C>,
        scales: &ViewCompilationArg<C>,
        scheme: QuantScheme,
        builder: &mut KernelBuilder,
    ) -> ViewExpand<Line<F>, C> {
        let values = View::<Line<Q>, C>::expand(values, builder);
        let scales = View::<S, C>::expand(scales, builder);
        let view = QuantizedViewExpand {
            values,
            scales,
            scheme,
            _ty: PhantomData,
        };
        ViewExpand::new(view)
    }

    #[allow(clippy::missing_transmute_annotations)]
    unsafe {
        match E::as_type(&builder.scope) {
            StorageType::Scalar(ElemType::Float(ty)) => match ty {
                FloatKind::F16 => t(expand_dynamic_f::<f16, C>(values, scales, scheme, builder)),
                FloatKind::BF16 => t(expand_dynamic_f::<bf16, C>(values, scales, scheme, builder)),
                FloatKind::Flex32 => t(expand_dynamic_f::<flex32, C>(
                    values, scales, scheme, builder,
                )),
                FloatKind::F32 => t(expand_dynamic_f::<f32, C>(values, scales, scheme, builder)),
                FloatKind::TF32 => t(expand_dynamic_f::<tf32, C>(values, scales, scheme, builder)),
                FloatKind::F64 => t(expand_dynamic_f::<f64, C>(values, scales, scheme, builder)),
                FloatKind::E2M1
                | FloatKind::E2M3
                | FloatKind::E3M2
                | FloatKind::E4M3
                | FloatKind::E5M2
                | FloatKind::UE8M0 => unreachable!("Minifloats don't implement `Float` ops"),
            },
            _ => unreachable!("Quantized view should only be used with floats"),
        }
    }
}
