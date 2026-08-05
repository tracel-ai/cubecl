use std::marker::PhantomData;

use super::*;
use crate::tensor::{
    View, ViewExpand, ViewOperations, ViewOperationsExpand,
    launch::{ViewArg, ViewCompilationArg},
    layout::{Coordinates, Coords1d},
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
    'a,
    Q: Scalar,
    NQ: Size,
    S: Scalar,
    F: Numeric,
    NF: Size,
    C: Coordinates + 'static,
> {
    values: View<'a, Vector<Q, NQ>, C>,
    scales: View<'a, S, C>,
    /// Per-tensor scale of a two-level scheme, already read and widened to f32.
    global: ComptimeOption<f32>,
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedView<'a, Q, NQ, S, F, NF, C>
{
    pub fn new(
        values: View<'a, Vector<Q, NQ>, C>,
        scales: View<'a, S, C>,
        global: ComptimeOption<f32>,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, Q, NQ, S, F, NF, C> {
            values,
            scales,
            global,
            scheme,
            _ty: PhantomData,
        }
    }
}

impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedView<'a, Q, NQ, S, F, NF, C>
{
    pub fn view(self) -> View<'a, Vector<F, NF>, C> {
        unexpanded!()
    }

    pub fn __expand_view(
        scope: &Scope,
        this: QuantizedViewExpand<'a, Q, NQ, S, F, NF, C>,
    ) -> ViewExpand<'a, Vector<F, NF>, C> {
        this.__expand_view_method(scope)
    }
}

impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedViewExpand<'a, Q, NQ, S, F, NF, C>
{
    pub fn new(
        values: ViewExpand<'a, Vector<Q, NQ>, C>,
        scales: ViewExpand<'a, S, C>,
        global: Option<NativeExpand<f32>>,
        scheme: QuantScheme,
    ) -> Self {
        QuantizedViewExpand::<'a, Q, NQ, S, F, NF, C> {
            values,
            scales,
            global: match global {
                Some(global) => ComptimeOptionExpand::Some(global),
                None => ComptimeOptionExpand::None,
            },
            scheme,
            _ty: PhantomData,
        }
    }

    pub fn __expand_view_method(self, scope: &Scope) -> ViewExpand<'a, Vector<F, NF>, C> {
        ViewExpand::new(scope, self)
    }
}

impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static> Vectorized
    for QuantizedView<'a, Q, NQ, S, F, NF, C>
{
}
impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    VectorizedExpand for QuantizedViewExpand<'a, Q, NQ, S, F, NF, C>
{
    fn vector_size(&self) -> VectorSize {
        self.values.vector_size() * self.scheme.num_quants()
    }
}

impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperations<Vector<F, NF>, C> for QuantizedView<'a, Q, NQ, S, F, NF, C>
{
}

impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    ViewOperationsExpand<Vector<F, NF>, C> for QuantizedViewExpand<'a, Q, NQ, S, F, NF, C>
{
    fn __expand_read_method(
        &self,
        scope: &Scope,
        pos: <C>::ExpandType,
    ) -> NativeExpand<Vector<F, NF>> {
        let value = self.values.clone().__expand_read_method(scope, pos.clone());
        let scale = self.scales.clone().__expand_read_method(scope, pos);

        dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.global, self.scheme)
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
        let scale = self.scales.clone().__expand_read_checked_method(scope, pos);

        dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.global, self.scheme)
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

        let value = dequantize_aligned::expand::<Q, S, F, NQ, NF>(
            scope,
            value,
            scale,
            self.global,
            self.scheme,
        );
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

        dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.global, self.scheme)
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

/// Storage (values) vector size: the float vector size divided by `num_quants`. Asserts the float
/// vector size is a multiple of `num_quants`, so a violation reports clearly here, not a cryptic cast error.
fn quant_vector_size_q(vector_size: usize, num_quants: usize) -> usize {
    assert!(
        vector_size >= num_quants && vector_size.is_multiple_of(num_quants),
        "quantized view float vector size {vector_size} must be a positive multiple of num_quants {num_quants}"
    );
    vector_size / num_quants
}

/// Read the per-tensor scale into the scope the view is built in, and widen it to f32.
///
/// One read for the whole kernel: the scale is a single value for the entire tensor, and a read
/// per element would be a global load the optimizer cannot hoist back out of a loop. Widening
/// here is what lets the level pick any param for it, and is also what keeps the two levels
/// multiplying in f32 later, since a block scale alone can overflow a narrow `F`.
fn expand_global_scale(
    global: &ViewCompilationArg<Coords1d>,
    param: QuantParam,
    builder: &mut KernelBuilder,
) -> NativeExpand<f32> {
    fn read<G: Scalar>(
        global: &ViewCompilationArg<Coords1d>,
        builder: &mut KernelBuilder,
    ) -> NativeExpand<f32> {
        let view = View::<G, Coords1d>::expand(global, builder);
        let pos = usize::__expand_cast_from(&builder.scope, 0.into());
        let scale = view.__expand_read_method(&builder.scope, pos);
        f32::__expand_cast_from(&builder.scope, scale)
    }

    match param {
        QuantParam::F32 => read::<f32>(global, builder),
        QuantParam::F16 => read::<f16>(global, builder),
        QuantParam::BF16 => read::<bf16>(global, builder),
        QuantParam::UE8M0 => read::<ue8m0>(global, builder),
        QuantParam::UE4M3 => read::<e4m3>(global, builder),
    }
}

struct ExpandDynamic<'a, E: Numeric, N: Size, C: Coordinates + 'static> {
    values: &'a ViewCompilationArg<C>,
    scales: &'a ViewCompilationArg<C>,
    global: Option<&'a ViewCompilationArg<Coords1d>>,
    scheme: QuantScheme,
    builder: &'a mut KernelBuilder,
    _ty: PhantomData<(E, N)>,
}

impl<'a, E: Numeric, N: Size, C: Coordinates + 'static> RunWithQuantType
    for ExpandDynamic<'a, E, N, C>
{
    type Output = ViewExpand<'static, Vector<E, N>, C>;

    fn execute<Q: Scalar, S: Scalar>(self) -> Self::Output {
        define_size!(NQ);

        check_global_bindings(self.scheme.level, self.global.is_some());

        let vector_size = N::__expand_value(&self.builder.scope);
        let vector_size_q = quant_vector_size_q(vector_size, self.scheme.num_quants());
        self.builder.scope.register_size::<NQ>(vector_size_q);

        let values = View::<Vector<Q, NQ>, C>::expand(self.values, self.builder);
        let scales = View::<S, C>::expand(self.scales, self.builder);
        // The check above pairs the binding with the level, so zipping drops neither.
        let global = self
            .global
            .zip(self.scheme.level.global_param())
            .map(|(global, param)| expand_global_scale(global, param, self.builder));
        let view = QuantizedViewExpand::new(values, scales, global, self.scheme);
        ViewExpand::new(&self.builder.scope, view)
    }
}

/// Register the per-tensor scale under the param the level stores it in, matching the element
/// type [`expand_global_scale`] reads it back with.
fn register_global_scale<R: Runtime>(
    global: ViewArg<Coords1d, R>,
    param: QuantParam,
    launcher: &mut KernelLauncher<R>,
) -> ViewCompilationArg<Coords1d> {
    match param {
        QuantParam::F32 => View::<f32, Coords1d>::register(global, launcher),
        QuantParam::F16 => View::<f16, Coords1d>::register(global, launcher),
        QuantParam::BF16 => View::<bf16, Coords1d>::register(global, launcher),
        QuantParam::UE8M0 => View::<ue8m0, Coords1d>::register(global, launcher),
        QuantParam::UE4M3 => View::<e4m3, Coords1d>::register(global, launcher),
    }
}

pub(crate) struct RegisterDynamic<'a, E: CubePrimitive, C: Coordinates + 'static, R: Runtime> {
    pub values: ViewArg<C, R>,
    pub scales: ViewArg<C, R>,
    pub global: Option<ViewArg<Coords1d, R>>,
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

        // Caught again on the dequantization path, but reporting it here names the launch that
        // asked for it rather than a kernel being expanded.
        check_global_bindings(self.scheme.level, self.global.is_some());

        self.launcher.with_scope(|scope| {
            let vector_size_q =
                quant_vector_size_q(E::__expand_vector_size(scope), self.scheme.num_quants());
            scope.register_size::<NQ>(vector_size_q);
        });

        let values = View::<Vector<Q, NQ>, C>::register(self.values, self.launcher);
        let scales = View::<S, C>::register(self.scales, self.launcher);
        let global = self
            .global
            .zip(self.scheme.level.global_param())
            .map(|(global, param)| Box::new(register_global_scale(global, param, self.launcher)));
        ViewCompilationArg::Quantized {
            values: Box::new(values),
            scales: Box::new(scales),
            global,
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
pub(crate) fn expand_dynamic<E: CubePrimitive, C: Coordinates + 'static>(
    values: &ViewCompilationArg<C>,
    scales: &ViewCompilationArg<C>,
    global: Option<&ViewCompilationArg<Coords1d>>,
    scheme: QuantScheme,
    builder: &mut KernelBuilder,
) -> ViewExpand<'static, E, C> {
    use core::mem::transmute as t;

    // To specify tighter trait bounds
    fn expand_dynamic_f<F: Numeric, NF: Size, C: Coordinates + 'static>(
        values: &ViewCompilationArg<C>,
        scales: &ViewCompilationArg<C>,
        global: Option<&ViewCompilationArg<Coords1d>>,
        scheme: QuantScheme,
        builder: &mut KernelBuilder,
    ) -> ViewExpand<'static, Vector<F, NF>, C> {
        let func = ExpandDynamic {
            values,
            scales,
            global,
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
                    values, scales, global, scheme, builder,
                )),
                FloatKind::BF16 => t(expand_dynamic_f::<bf16, NF, C>(
                    values, scales, global, scheme, builder,
                )),
                FloatKind::Flex32 => t(expand_dynamic_f::<flex32, NF, C>(
                    values, scales, global, scheme, builder,
                )),
                FloatKind::F32 => t(expand_dynamic_f::<f32, NF, C>(
                    values, scales, global, scheme, builder,
                )),
                FloatKind::TF32 => t(expand_dynamic_f::<tf32, NF, C>(
                    values, scales, global, scheme, builder,
                )),
                FloatKind::F64 => t(expand_dynamic_f::<f64, NF, C>(
                    values, scales, global, scheme, builder,
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

#[cfg(test)]
mod tests {
    use super::{RunWithQuantType, quant_vector_size_q, run_with_quant_type};
    use cubecl_common::quant::scheme::{QuantLevel, QuantParam, QuantScheme};
    use cubecl_core::prelude::Scalar;

    struct Dispatched;

    impl RunWithQuantType for Dispatched {
        type Output = bool;

        fn execute<Q: Scalar, S: Scalar>(self) -> bool {
            true
        }
    }

    fn two_level_scheme(global: QuantParam) -> QuantScheme {
        QuantScheme::default().with_level(QuantLevel::block_tensor([32], global))
    }

    #[test]
    fn one_level_scheme_dispatches() {
        assert!(run_with_quant_type(Dispatched, QuantScheme::default()));
    }

    /// The per-tensor scale is read through a binding of its own, so a level that has one does not
    /// change how the value and block scale types dispatch.
    #[test]
    fn two_level_scheme_dispatches() {
        assert!(run_with_quant_type(
            Dispatched,
            two_level_scheme(QuantParam::F32)
        ));
        assert!(run_with_quant_type(
            Dispatched,
            two_level_scheme(QuantParam::UE4M3)
        ));
    }

    #[test]
    fn vector_size_q_exact_multiple() {
        assert_eq!(quant_vector_size_q(8, 8), 1);
        assert_eq!(quant_vector_size_q(16, 8), 2);
        assert_eq!(quant_vector_size_q(16, 16), 1);
    }

    #[test]
    #[should_panic(expected = "positive multiple of num_quants")]
    fn vector_size_q_non_multiple_panics() {
        let _ = quant_vector_size_q(8, 16);
    }
}
