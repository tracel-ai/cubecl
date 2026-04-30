use std::marker::PhantomData;

use super::*;
use crate::tensor::{
    View, ViewExpand, ViewOperations, ViewOperationsExpand,
    launch::{ViewArg, ViewCompilationArg},
    layout::Coordinates,
};
use cubecl::prelude::*;
use cubecl_common::{
    e2m1x2, e4m3, e5m2,
    quant::scheme::{QuantParam, QuantScheme, QuantStore, QuantValue},
    ue8m0,
};
use cubecl_core::{
    self as cubecl, define_size,
    ir::{ElemType, FloatKind, StorageType, VectorSize},
    prelude::barrier::Barrier,
    unexpanded,
};
use half::{bf16, f16};

/// View that dequantizes after loads. Scales layout should take values coordinates and map them
/// to the corresponding scale.
///
/// # Warning
/// Assumes only one scale maps to a single load. Adjust vector size of values or block size to ensure
/// this.
/// Must ensure `block_size.is_multiple_of(vector_size * scheme.num_quants())`.
#[expect(dead_code, reason = "only used in expand")]
#[derive(CubeType, CubeLaunch, Clone)]
pub struct QuantizedView<
    Q: Scalar,
    NQ: Size,
    S: Scalar,
    F: Numeric,
    NF: Size,
    C: Coordinates + 'static,
> {
    values: View<Vector<Q, NQ>, C>,
    scales: View<S, C>,
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedView<Q, NQ, S, F, NF, C>
{
    pub fn new(
        values: View<Vector<Q, NQ>, C>,
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

impl<Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedView<Q, NQ, S, F, NF, C>
{
    pub fn view(self) -> View<Vector<F, NF>, C> {
        unexpanded!()
    }

    pub fn __expand_view(
        scope: &Scope,
        this: QuantizedViewExpand<Q, NQ, S, F, NF, C>,
    ) -> ViewExpand<Vector<F, NF>, C, ReadOnly> {
        this.__expand_view_method(scope)
    }
}

impl<Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedViewExpand<Q, NQ, S, F, NF, C>
{
    pub fn new(
        values: ViewExpand<Vector<Q, NQ>, C>,
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

    pub fn __expand_view_method(self, _scope: &Scope) -> ViewExpand<Vector<F, NF>, C, ReadOnly> {
        ViewExpand::new(self)
    }
}

impl<Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static> Vectorized
    for QuantizedView<Q, NQ, S, F, NF, C>
{
}
impl<Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    VectorizedExpand for QuantizedViewExpand<Q, NQ, S, F, NF, C>
{
    fn vector_size(&self) -> VectorSize {
        self.values.vector_size() * self.scheme.num_quants()
    }
}

impl<Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperations<Vector<F, NF>, C> for QuantizedView<Q, NQ, S, F, NF, C>
{
}

impl<Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperationsExpand<Vector<F, NF>, C> for QuantizedViewExpand<Q, NQ, S, F, NF, C>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        let value = self.values.clone().__expand_read_method(scope, pos.clone());
        let scale = self.scales.clone().__expand_read_method(scope, pos);

        dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.scheme)
    }

    fn __expand_read_checked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
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
        scope: &Scope,
        pos: <C>::ExpandType,
        mask_value: NativeExpand<Vector<F, NF>>,
    ) -> NativeExpand<Vector<F, NF>> {
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
        select::expand::<Vector<F, NF>>(scope, in_bounds, value, mask_value)
    }

    fn __expand_read_unchecked_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
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

    fn __expand_as_linear_slice_method(
        &self,
        _scope: &Scope,
        _pos: <C>::ExpandType,
        _end: <C>::ExpandType,
    ) -> &SliceExpand<Vector<F, NF>> {
        panic!("Can't create raw slice for quantized view")
    }

    fn __expand_shape_method(&self, scope: &Scope) -> <C>::ExpandType {
        self.values.clone().__expand_shape_method(scope)
    }

    fn __expand_is_in_bounds_method(
        &self,
        scope: &Scope,
        pos: C::ExpandType,
    ) -> NativeExpand<bool> {
        self.values.clone().__expand_is_in_bounds_method(scope, pos)
    }

    fn __expand_tensor_map_load_method(
        &self,
        _scope: &Scope,
        _barrier: &NativeExpand<Barrier>,
        _shared_memory: &mut SliceExpand<Vector<F, NF>>,
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
    type Output = ViewExpand<Vector<E, N>, C>;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output {
        define_size!(NQ);

        let vector_size = N::__expand_value(&self.builder.scope);
        let vector_size_q = vector_size / self.scheme.num_quants();
        self.builder.scope.register_size::<NQ>(vector_size_q);

        let values = View::<Vector<Q, NQ>, C>::expand(self.values, self.builder);
        let scales = View::<S, C>::expand(self.scales, self.builder);
        let view = QuantizedViewExpand::new(values, scales, self.scheme);
        ViewExpand::new(view)
    }
}

pub(crate) struct RegisterDynamic<'a, E: CubePrimitive, C: Coordinates + 'static, R: Runtime> {
    pub values: ViewArg<C, R>,
    pub scales: ViewArg<C, R>,
    pub scheme: QuantScheme,
    pub launcher: &'a mut KernelLauncher<R>,
    pub _ty: PhantomData<E>,
}

impl<'a, E: CubePrimitive, C: Coordinates + 'static, R: Runtime> RunWithQuantType
    for RegisterDynamic<'a, E, C, R>
{
    type Output = ViewCompilationArg<C>;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output {
        define_size!(NQ);

        self.launcher.with_scope(|scope| {
            let vector_size_q = E::__expand_vector_size(scope) / self.scheme.num_quants();
            scope.register_size::<NQ>(vector_size_q);
        });

        let values = View::<Vector<Q, NQ>, C>::register(self.values, self.launcher);
        let scales = View::<S, C>::register(self.scales, self.launcher);
        ViewCompilationArg::Quantized {
            values: Box::new(values),
            scales: Box::new(scales),
            scheme: self.scheme,
        }
    }
}

/// Run a function with the quantization storage type and scale. Useful when concrete types are
/// required but aren't available, and only the dynamic schema is known.
pub fn run_with_quant_type<F: RunWithQuantType>(func: F, scheme: QuantScheme) -> F::Output {
    fn run_with_q<F: RunWithQuantType, Q: Scalar>(func: F, scheme: QuantScheme) -> F::Output {
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
    ) -> ViewExpand<Vector<F, NF>, C> {
        let func = ExpandDynamic {
            values,
            scales,
            scheme,
            builder,
            _ty: PhantomData::<(F, NF)>,
        };
        run_with_quant_type(func, scheme)
    }

    define_size!(NF);

    let vector_size = E::__expand_vector_size(&builder.scope);

    builder.scope.register_size::<NF>(vector_size);

    #[allow(clippy::missing_transmute_annotations)]
    unsafe {
        match E::__expand_as_type(&builder.scope).storage_type() {
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
