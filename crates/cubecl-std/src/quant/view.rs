use std::marker::PhantomData;

use super::*;
use crate::tensor::{
    View, ViewExpand, ViewOperations, ViewOperationsExpand,
    launch::{ScaleBindings, ScaleBindingsCompilationArg, ViewArg, ViewCompilationArg},
    layout::Coordinates,
};
use cubecl::prelude::*;
use cubecl_common::{
    e2m1x2, e4m3, e5m2,
    quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
    ue8m0,
};
use cubecl_core::{
    self as cubecl, define_size,
    ir::{ElemType, FloatKind, VectorSize},
    prelude::barrier::Barrier,
    unexpanded,
};
use half::{bf16, f16};

/// The part of each read's scale the caller already knew when it built the view, held in one
/// register rather than read per position; whatever is not known up front is read through the
/// scales view. The discriminant is comptime, so each variant compiles its own kernel with
/// nothing of the others in it.
#[derive(Clone, Copy, CubeType, CubeLaunch)]
#[expand(derive(Clone, Copy))]
pub enum KnownScale {
    /// Nothing known up front: each read looks its whole scale up at its position.
    None,
    /// The per-tensor scale of a two-level scheme; each read still looks its block scale up and
    /// multiplies this in.
    Global(f32),
    /// The whole scale, whatever the caller multiplied into it. The scales view is never read:
    /// its address arithmetic and its load leave the kernel.
    Whole(f32),
}

#[cube]
impl KnownScale {
    /// The scale a value dequantizes against once `scale`, looked up for its position, meets what
    /// this register holds.
    pub fn effective(&self, scale: f32) -> f32 {
        #[comptime]
        match self {
            KnownScale::None => scale,
            KnownScale::Global(global) => global * scale,
            KnownScale::Whole(whole) => *whole,
        }
    }
}

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
    known_scale: KnownScale,
    #[cube(comptime)]
    scheme: QuantScheme,
    #[cube(comptime)]
    _ty: PhantomData<(F, NF)>,
}

#[cube]
impl<'a, Q: Scalar, NQ: Size, S: Scalar, F: Numeric, NF: Size, C: Coordinates + 'static>
    QuantizedView<'a, Q, NQ, S, F, NF, C>
{
    /// A view reading every scale per position through `scales`.
    ///
    /// Takes a one-level scheme: the per-tensor scale of a two-level one never rides a signature
    /// here, it is either bound at launch or already multiplied in by a caller using
    /// [`new_with_whole_scale`](Self::new_with_whole_scale).
    pub fn new(
        values: View<'a, Vector<Q, NQ>, C>,
        scales: View<'a, S, C>,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        comptime!(crate::quant::check_scale_bindings(&scheme, 1));
        QuantizedView::<'a, Q, NQ, S, F, NF, C> {
            values,
            scales,
            known_scale: KnownScale::new_None(),
            scheme,
            _ty: PhantomData,
        }
    }

    /// [`new`](Self::new) with the per-tensor scale already read into `global_scale`: each read
    /// still looks its block scale up through `scales`, then multiplies the register in. Takes a
    /// two-level scheme, whose per-tensor level the register stands for; a one-level scheme reads
    /// through [`new`](Self::new).
    pub fn new_with_global_scale(
        values: View<'a, Vector<Q, NQ>, C>,
        scales: View<'a, S, C>,
        global_scale: f32,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, Q, NQ, S, F, NF, C> {
            values,
            scales,
            known_scale: KnownScale::new_Global(global_scale),
            scheme,
            _ty: PhantomData,
        }
    }

    /// [`new`](Self::new) for values that share one scale: `scale` is the whole scale for every
    /// value read through this view, whatever levels the caller multiplied into it, so the scales
    /// view rides along unread. Only a caller that knows the values it will read share a block
    /// can say so, which is why this exists on the cube side.
    pub fn new_with_whole_scale(
        values: View<'a, Vector<Q, NQ>, C>,
        scales: View<'a, S, C>,
        scale: f32,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, Q, NQ, S, F, NF, C> {
            values,
            scales,
            known_scale: KnownScale::new_Whole(scale),
            scheme,
            _ty: PhantomData,
        }
    }

    /// [`new`](Self::new) with whatever the caller already knows of the scale, in whichever
    /// [`KnownScale`] form it holds it. Reads assert the register agrees with the scheme.
    pub fn new_with_known_scale(
        values: View<'a, Vector<Q, NQ>, C>,
        scales: View<'a, S, C>,
        known_scale: KnownScale,
        #[comptime] scheme: QuantScheme,
    ) -> Self {
        QuantizedView::<'a, Q, NQ, S, F, NF, C> {
            values,
            scales,
            known_scale,
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
        known_scale: KnownScaleExpand,
        scheme: QuantScheme,
    ) -> Self {
        QuantizedViewExpand::<'a, Q, NQ, S, F, NF, C> {
            values,
            scales,
            known_scale,
            scheme,
            _ty: PhantomData,
        }
    }

    /// Dequantize `value` with the effective scale this view assigns it, reading the per-position
    /// scale through `read_scale` unless the register holds the whole scale.
    fn dequant(
        &self,
        scope: &Scope,
        value: NativeExpand<Vector<Q, NQ>>,
        read_scale: impl FnOnce(&Scope) -> NativeExpand<S>,
    ) -> NativeExpand<Vector<F, NF>> {
        // The reading variants are where the register can disagree with the scheme: the static
        // constructors take a scheme without inspecting it. A whole scale stands for whatever the
        // caller multiplied into it, so the scheme says nothing about it.
        match self.known_scale {
            KnownScaleExpand::None => {
                assert!(
                    self.scheme.num_levels() == 1,
                    "every scale is read from the scales view, but {:?} has a per-tensor scale nothing multiplies in",
                    self.scheme,
                );
                let scale = read_scale(scope);
                dequantize_aligned::expand::<Q, S, F, NQ, NF>(scope, value, scale, self.scheme)
            }
            KnownScaleExpand::Global(global_scale) => {
                assert!(
                    self.scheme.num_levels() > 1,
                    "an global scale rides in a register, but {:?} has no per-tensor level over its blocks it could hold",
                    self.scheme,
                );
                check_global_levels(&self.scheme);
                let scale = read_scale(scope);
                let scale = multiply_global_scale::expand::<S>(scope, global_scale, scale);
                dequantize_aligned_wide::expand::<Q, F, NQ, NF>(scope, value, scale, self.scheme)
            }
            KnownScaleExpand::Whole(scale) => {
                dequantize_aligned_wide::expand::<Q, F, NQ, NF>(scope, value, scale, self.scheme)
            }
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
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        self.values.__expand_vector_size_method(scope) * self.scheme.num_quants()
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
        self.dequant(scope, value, |scope| {
            self.scales.clone().__expand_read_method(scope, pos)
        })
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
        self.dequant(scope, value, |scope| {
            self.scales.clone().__expand_read_checked_method(scope, pos)
        })
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
        let in_bounds = self.__expand_is_in_bounds_method(scope, pos.clone());

        let value = self.dequant(scope, value, |scope| {
            self.scales.clone().__expand_read_checked_method(scope, pos)
        });
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
        self.dequant(scope, value, |scope| {
            self.scales
                .clone()
                .__expand_read_unchecked_method(scope, pos)
        })
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

/// Register the per-tensor scale binding. Registered as f32 to match the element type
/// [`expand_known_scale`] reads it back with.
fn register_global_scale<R: Runtime>(
    global_scale: Option<BufferArg<R>>,
    launcher: &mut KernelLauncher<R>,
) -> Option<BufferCompilationArg> {
    global_scale.map(|global_scale| <[f32] as LaunchArg>::register(global_scale, launcher))
}

/// The known scale a launch's bindings expand to.
///
/// An global scale is read once for the whole kernel: it is a single value for the entire tensor,
/// and a read per element would be a global load the optimizer cannot hoist back out of a loop.
/// Reading it as f32 is what keeps the two scales multiplying in f32 later, since a block scale
/// alone can overflow a narrow `F`.
fn expand_known_scale(
    global_scale: Option<&BufferCompilationArg>,
    builder: &mut KernelBuilder,
) -> KnownScaleExpand {
    match global_scale {
        Some(global_scale) => {
            let buffer = <[f32] as LaunchArg>::expand(global_scale, builder);
            let pos = NativeExpand::<usize>::from_lit(&builder.scope, 0);
            KnownScaleExpand::Global(*buffer.__expand_index_method(&builder.scope, pos))
        }
        None => KnownScaleExpand::None,
    }
}

struct ExpandDynamic<'a, E: Numeric, N: Size, C: Coordinates + 'static> {
    values: &'a ViewCompilationArg<C>,
    scales: &'a ScaleBindingsCompilationArg<C>,
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

        let vector_size = N::__expand_value(&self.builder.scope);
        let vector_size_q = quant_vector_size_q(vector_size, self.scheme.num_quants());
        self.builder.scope.register_size::<NQ>(vector_size_q);

        check_scale_bindings(&self.scheme, self.scales.len());

        let values = View::<Vector<Q, NQ>, C>::expand(self.values, self.builder);
        let scales = View::<S, C>::expand(&self.scales.inner, self.builder);
        let known_scale = expand_known_scale(self.scales.global_scale.as_ref(), self.builder);
        let view = QuantizedViewExpand::new(values, scales, known_scale, self.scheme);
        ViewExpand::new(&self.builder.scope, view)
    }
}

pub(crate) struct RegisterDynamic<'a, E: CubePrimitive, C: Coordinates + 'static, R: Runtime> {
    pub values: ViewArg<C, R>,
    pub scales: ScaleBindings<C, R>,
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
            let vector_size_q =
                quant_vector_size_q(E::__expand_vector_size(scope), self.scheme.num_quants());
            scope.register_size::<NQ>(vector_size_q);
        });

        check_scale_bindings(&self.scheme, self.scales.len());

        let values = View::<Vector<Q, NQ>, C>::register(self.values, self.launcher);
        let inner = View::<S, C>::register(*self.scales.inner, self.launcher);
        let global_scale = register_global_scale(self.scales.global_scale, self.launcher);
        ViewCompilationArg::Quantized {
            values: Box::new(values),
            scales: ScaleBindingsCompilationArg {
                inner: Box::new(inner),
                global_scale,
            },
            scheme: self.scheme,
        }
    }
}

/// Run a function with the quantization storage type and scale. Useful when concrete types are
/// required but aren't available, and only the dynamic schema is known.
pub fn run_with_quant_type<F: RunWithQuantType>(func: F, scheme: QuantScheme) -> F::Output {
    fn run_with_q<F: RunWithQuantType, Q: Scalar>(func: F, scheme: QuantScheme) -> F::Output {
        match scheme.scale_dtype() {
            ScaleDtype::F32 => func.execute::<Q, f32>(),
            ScaleDtype::F16 => func.execute::<Q, f16>(),
            ScaleDtype::BF16 => func.execute::<Q, bf16>(),
            ScaleDtype::UE8M0 => func.execute::<Q, ue8m0>(),
            ScaleDtype::UE4M3 => func.execute::<Q, e4m3>(),
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
    scales: &ScaleBindingsCompilationArg<C>,
    scheme: QuantScheme,
    builder: &mut KernelBuilder,
) -> ViewExpand<'static, E, C> {
    use core::mem::transmute as t;

    // To specify tighter trait bounds
    fn expand_dynamic_f<F: Numeric, NF: Size, C: Coordinates + 'static>(
        values: &ViewCompilationArg<C>,
        scales: &ScaleBindingsCompilationArg<C>,
        scheme: QuantScheme,
        builder: &mut KernelBuilder,
    ) -> ViewExpand<'static, Vector<F, NF>, C> {
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

    let vector_size = E::__expand_vector_size(builder);

    builder.scope.register_size::<NF>(vector_size);

    #[allow(clippy::missing_transmute_annotations)]
    unsafe {
        match E::Scalar::elem_type(builder) {
            ElemType::Float(ty) => match ty {
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
                | FloatKind::E2M1x2
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
    use cubecl_common::quant::scheme::{QuantScheme, ScaleDtype};
    use cubecl_core::prelude::Scalar;

    struct Dispatched;

    impl RunWithQuantType for Dispatched {
        type Output = bool;

        fn execute<Q: Scalar, S: Scalar>(self) -> bool {
            true
        }
    }

    #[test]
    fn one_level_scheme_dispatches() {
        assert!(run_with_quant_type(Dispatched, QuantScheme::default()));
    }

    /// The per-tensor scale is read through a binding of its own, so it does not change how the
    /// value and block scale types dispatch.
    #[test]
    fn two_level_scheme_dispatches() {
        let scheme = QuantScheme::default()
            .per_block([32], ScaleDtype::F32)
            .per_tensor(ScaleDtype::F32);
        assert!(run_with_quant_type(Dispatched, scheme));
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
