use std::marker::PhantomData;

use super::*;
use crate::tensor::{
    View, ViewExpand, ViewOperations, ViewOperationsExpand, launch::ViewCompilationArg,
    layout::Coordinates,
};
use cubecl::prelude::*;
use cubecl_common::{
    e2m1x2, e4m3, e5m2,
    quant::scheme::{QuantParam, QuantScheme, QuantStore, QuantValue},
    ue8m0,
};
use cubecl_core::{
    self as cubecl,
    ir::{ElemType, FloatKind, LineSize, StorageType},
    prelude::barrier::BarrierExpand,
    unexpanded,
};
use half::{bf16, f16};

/// View that dequantizes after loads. Scales layout should take values coordinates and map them
/// to the corresponding scale.
///
/// # Warning
/// Assumes only one scale maps to a single load. Adjust line size of values or block size to ensure
/// this.
/// Must ensure `block_size.is_multiple_of(line_size * scheme.num_quants())`.
#[expect(dead_code, reason = "only used in expand")]
#[derive(CubeType, CubeLaunch, Clone, Copy)]
pub struct QuantizedView<
    Q: CubePrimitive,
    NQ: Size,
    S: CubePrimitive,
    F: Numeric,
    NF: Size,
    C: Coordinates + 'static,
> {
    values: View<Line<Q, NQ>, C>,
    scales: View<S, C>,
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<Q: CubePrimitive, NQ: Size, S: CubePrimitive, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedView<Q, NQ, S, F, NF, C>
{
    pub fn new(
        values: View<Line<Q, NQ>, C>,
        scales: View<S, C>,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<Q, NQ, S, F, NF, C> {
            values,
            scales,
            scheme,
            _ty: PhantomData,
        }
    }
}

impl<Q: CubePrimitive, NQ: Size, S: CubePrimitive, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedView<Q, NQ, S, F, NF, C>
{
    pub fn view(self) -> View<Line<F, NF>, C> {
        unexpanded!()
    }

    pub fn __expand_view(
        scope: &mut Scope,
        this: QuantizedViewExpand<Q, NQ, S, F, NF, C>,
    ) -> ViewExpand<Line<F, NF>, C, ReadOnly> {
        this.__expand_view_method(scope)
    }
}

impl<Q: CubePrimitive, NQ: Size, S: CubePrimitive, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedViewExpand<Q, NQ, S, F, NF, C>
{
    pub fn new(
        values: ViewExpand<Line<Q, NQ>, C>,
        scales: ViewExpand<S, C>,
        scheme: QuantScheme,
    ) -> Self {
        QuantizedViewExpand::<Q, NQ, S, F, NF, C> {
            values,
            scales,
            scheme,
            _ty: PhantomData,
        }
    }

    pub fn __expand_view_method(self, _scope: &mut Scope) -> ViewExpand<Line<F, NF>, C, ReadOnly> {
        ViewExpand::new(self)
    }
}

impl<Q: CubePrimitive, NQ: Size, S: CubePrimitive, F: Numeric, NF: Size, C: Coordinates + 'static>
    Lined for QuantizedView<Q, NQ, S, F, NF, C>
{
}
impl<Q: CubePrimitive, NQ: Size, S: CubePrimitive, F: Numeric, NF: Size, C: Coordinates + 'static>
    LinedExpand for QuantizedViewExpand<Q, NQ, S, F, NF, C>
{
    fn line_size(&self) -> LineSize {
        self.values.line_size() * self.scheme.num_quants()
    }
}

impl<Q: CubePrimitive, NQ: Size, S: CubePrimitive, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperations<Line<F, NF>, C> for QuantizedView<Q, NQ, S, F, NF, C>
{
}

impl<Q: CubePrimitive, NQ: Size, S: CubePrimitive, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperationsExpand<Line<F, NF>, C> for QuantizedViewExpand<Q, NQ, S, F, NF, C>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> ExpandElementTyped<Line<F, NF>> {
        let value = self.values.clone().__expand_read_method(scope, pos.clone());
        let scale = self.scales.clone().__expand_read_method(scope, pos);

        dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.scheme)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> ExpandElementTyped<Line<F, NF>> {
        let value = self
            .values
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let scale = self
            .scales
            .clone()
            .__expand_read_checked_method(scope, pos.clone());

        dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.scheme)
    }

    fn __expand_read_masked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
        mask_value: ExpandElementTyped<Line<F, NF>>,
    ) -> ExpandElementTyped<Line<F, NF>> {
        let value = self
            .values
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let scale = self
            .scales
            .clone()
            .__expand_read_checked_method(scope, pos.clone());
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos);

        let value = dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.scheme);
        select::expand::<Line<F, NF>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &mut Scope,
        pos: <C>::ExpandType,
    ) -> ExpandElementTyped<Line<F, NF>> {
        let value = self
            .values
            .clone()
            .__expand_read_unchecked_method(scope, pos.clone());
        let scale = self
            .scales
            .clone()
            .__expand_read_unchecked_method(scope, pos);

        dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.scheme)
    }

    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _pos: <C>::ExpandType,
        _end: <C>::ExpandType,
    ) -> SliceExpand<Line<F, NF>, ReadOnly> {
        panic!("Can't create raw slice for quantized view")
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
        _shared_memory: SliceExpand<Line<F, NF>, ReadWrite>,
        _pos: C::ExpandType,
    ) {
        panic!("Can't use tensor map functions on quantized view");
    }
}

struct ExpandDynamic<'a, E: Numeric, N: Size, C: Coordinates + 'static> {
    values: &'a ViewCompilationArg<C>,
    scales: &'a ViewCompilationArg<C>,
    scheme: QuantScheme,
    builder: &'a mut KernelBuilder,
    _ty: PhantomData<(E, N)>,
}

impl<'a, E: Numeric, N: Size, C: Coordinates + 'static> RunWithQuantType
    for ExpandDynamic<'a, E, N, C>
{
    type Output = ViewExpand<Line<E, N>, C>;

    fn execute<Q: CubePrimitive, S: CubePrimitive>(self) -> Self::Output {
        #[derive(Clone, Copy)]
        struct NQ;
        impl Size for NQ {
            fn __expand_value(scope: &mut Scope) -> usize {
                scope.resolve_size::<NQ>().unwrap()
            }
        }

        self.builder
            .scope
            .register_size::<NQ>(self.values.line_size());

        let values = View::<Line<Q, NQ>, C>::expand(self.values, self.builder);
        let scales = View::<S, C>::expand(self.scales, self.builder);
        let view = QuantizedViewExpand::new(values, scales, self.scheme);
        ViewExpand::new(view)
    }
}

/// Run a function with the quantization storage type and scale. Useful when concrete types are
/// required but aren't available, and only the dynamic schema is known.
pub fn run_with_quant_type<F: RunWithQuantType>(func: F, scheme: QuantScheme) -> F::Output {
    fn run_with_q<F: RunWithQuantType, Q: CubePrimitive>(
        func: F,
        scheme: QuantScheme,
    ) -> F::Output {
        match scheme.param {
            QuantParam::F32 => func.execute::<Q, f32>(),
            QuantParam::F16 => func.execute::<Q, f16>(),
            QuantParam::BF16 => func.execute::<Q, bf16>(),
            QuantParam::UE8M0 => func.execute::<Q, ue8m0>(),
            QuantParam::UE4M3 => func.execute::<Q, e4m3>(),
        }
    }

    let run_q = match scheme.store {
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F => run_with_q::<F, i8>,
            QuantValue::Q8S => run_with_q::<F, i8>,
            QuantValue::E5M2 => run_with_q::<F, e5m2>,
            QuantValue::E4M3 => run_with_q::<F, e4m3>,
            QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S
            | QuantValue::E2M1 => {
                panic!("Sub-byte quantization can't be native")
            }
        },
        QuantStore::PackedU32(_) => run_with_q::<F, u32>,
        QuantStore::PackedNative(_) => run_with_q::<F, e2m1x2>,
    };
    run_q(func, scheme)
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
    fn expand_dynamic_f<F: Numeric, NF: Size, C: Coordinates + 'static>(
        values: &ViewCompilationArg<C>,
        scales: &ViewCompilationArg<C>,
        scheme: QuantScheme,
        builder: &mut KernelBuilder,
    ) -> ViewExpand<Line<F, NF>, C> {
        let func = ExpandDynamic {
            values,
            scales,
            scheme,
            builder,
            _ty: PhantomData::<(F, NF)>,
        };
        run_with_quant_type(func, scheme)
    }

    #[derive(Clone, Copy)]
    struct NF;
    impl Size for NF {
        fn __expand_value(scope: &mut Scope) -> usize {
            scope.resolve_size::<NF>().unwrap()
        }
    }

    builder
        .scope
        .register_size::<NF>(values.line_size() * scheme.num_quants());

    #[allow(clippy::missing_transmute_annotations)]
    unsafe {
        match E::as_type(&builder.scope) {
            StorageType::Scalar(ElemType::Float(ty)) => match ty {
                FloatKind::F16 => t(expand_dynamic_f::<f16, NF, C>(
                    values, scales, scheme, builder,
                )),
                FloatKind::BF16 => t(expand_dynamic_f::<bf16, NF, C>(
                    values, scales, scheme, builder,
                )),
                FloatKind::Flex32 => t(expand_dynamic_f::<flex32, NF, C>(
                    values, scales, scheme, builder,
                )),
                FloatKind::F32 => t(expand_dynamic_f::<f32, NF, C>(
                    values, scales, scheme, builder,
                )),
                FloatKind::TF32 => t(expand_dynamic_f::<tf32, NF, C>(
                    values, scales, scheme, builder,
                )),
                FloatKind::F64 => t(expand_dynamic_f::<f64, NF, C>(
                    values, scales, scheme, builder,
                )),
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
