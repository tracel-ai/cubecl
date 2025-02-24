use crate::tensor::{VirtualTensorOperations, VirtualTensorOperationsExpand};
use cubecl::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
/// Arguments for the matrix multiplication algorithm.
pub trait MatmulArgs: Send + Sync + 'static + Clone {
    /// Type used for the input.
    type Input<EG: Numeric>: LaunchArg + CubeType;
    /// Type used for the output.
    type Output<EG: Numeric>: LaunchArg + CubeType;
    /// Inner state that is used to create [tensor inputs](TensorInput) and
    /// [tensor outputs](TensorOutput) .
    type State<EG: Numeric>: CubeType;

    /// Init the state.
    fn init_state<EG: Numeric>(
        input: &Self::Input<EG>,
        output: &mut Self::Output<EG>,
    ) -> Self::State<EG>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_lhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_rhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG>;

    /// Read the line of the lhs tensor using the state at the given coordinate.
    fn read_window_lhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>>;
    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>>;

    /// Write the line to the output at the given coordinate using the state.
    fn write_out<EG: Numeric>(state: &mut Self::State<EG>, coordinate: u32, value: Line<EG>);

    /// Get the rank of the lhs tensor using the state.
    fn rank_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the rank of the rhs tensor using the state.
    fn rank_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32;
    /// Get the rank of the out tensor using the state.
    fn rank_out<EG: Numeric>(state: &Self::State<EG>) -> u32;

    /// Get the shape of the lhs tensor using the state.
    fn shape_lhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the shape of the rhs tensor using the state.
    fn shape_rhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the shape of the out tensor using the state.
    fn shape_out<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;

    /// Get the stride of the lhs tensor using the state.
    fn stride_lhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the stride of the rhs tensor using the state.
    fn stride_rhs<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;
    /// Get the stride of the out tensor using the state.
    fn stride_out<EG: Numeric>(state: &Self::State<EG>, axis: u32) -> u32;

    /// This should only be called after checking that the matmul is quantized.
    /// While the function allows for generic EG, it should only be called with u8.
    /// Unfortunately, I didn't find a clean way to express this in the type system.
    fn quantization<EG: Numeric>(state: &Self::State<EG>) -> Quantization;
}

#[derive(CubeType, Clone, Copy)]
pub struct Quantization {
    pub zero_offset_lhs: i32,
    pub zero_offset_rhs: i32,
    pub zero_offset_out: i32,
    scaling_multiplier: i64,
    scaling_rounding: i64,
    scaling_shift: i64,
}

#[cube]
impl Quantization {
    pub fn scale_approx(&self, values: Line<i32>) -> Line<i32> {
        // TODO can it overflow?
        let prod = Line::<i64>::cast_from(values) * Line::new(self.scaling_multiplier);
        let prod_with_rounding = prod + Line::new(self.scaling_rounding);
        Line::cast_from(prod_with_rounding >> Line::new(self.scaling_shift))
    }
}

#[derive(Clone, Copy)]
/// Identification of the [tensor input](TensorInput).
pub enum TensorInputIdent {
    Lhs,
    Rhs,
}

/// Tensor input representation.
///
/// You can use the tensor input as if it was a pointer to the actually tensor.
pub struct TensorInput<EG: Numeric, GA: MatmulArgs> {
    state: *const GA::State<EG>,
    ident: TensorInputIdent,
}

impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperations<EG> for TensorInput<EG, MA> {}
impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperations<EG> for TensorOutput<EG, MA> {}

impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<EG> for TensorOutputExpand<EG, MA> {
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<EG>> {
        panic!("Can't read output tensor");
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<u32>,
        _end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<EG>>> {
        panic!("Can't read output tensor");
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<Line<EG>>,
    ) {
        TensorOutputExpand::__expand_write_method(self.clone(), scope, index, value)
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorOutputExpand::__expand_rank_method(self.clone(), scope)
    }
}

impl<EG: Numeric, MA: MatmulArgs> VirtualTensorOperationsExpand<EG> for TensorInputExpand<EG, MA> {
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<EG>> {
        TensorInputExpand::__expand_read_method(self.clone(), scope, index)
    }
    fn __expand_read_window_method(
        &self,
        context: &mut Scope,
        start: ExpandElementTyped<u32>,
        end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<EG>>> {
        TensorInputExpand::__expand_read_window_method(self.clone(), context, start, end)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<EG>>,
    ) {
        panic!("Can't write to input tensor");
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_shape_method(self.clone(), scope, axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_stride_method(self.clone(), scope, axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        TensorInputExpand::__expand_rank_method(self.clone(), scope)
    }
}

/// Tensor output representation.
///
/// You can use the tensor output as if it was a pointer to the actually tensor.
///
/// # Warning
///
/// There is no mutability guarantee.
pub struct TensorOutput<EG: Numeric, GA: MatmulArgs> {
    state: *mut GA::State<EG>,
}

/// Expand type for [tensor input](TensorInput).
pub struct TensorInputExpand<EG: Numeric, GA: MatmulArgs> {
    state: <GA::State<EG> as CubeType>::ExpandType,
    ident: TensorInputIdent,
}

/// Expand type for [tensor output](TensorOutput).
pub struct TensorOutputExpand<EG: Numeric, GA: MatmulArgs> {
    state: <GA::State<EG> as CubeType>::ExpandType,
}

#[cube]
impl<EG: Numeric, MA: MatmulArgs> TensorInput<EG, MA> {
    /// Create a [tensor input](TensorInput) from the state and the [ident](TensorInputIdent).
    pub fn new(state: &MA::State<EG>, #[comptime] ident: TensorInputIdent) -> TensorInput<EG, MA> {
        TensorInput::<EG, MA> { state, ident }
    }

    //// Read the tensor at the given coordinate.
    pub fn read_window(&self, start: u32, end: u32) -> Slice<Line<EG>> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::read_window_lhs(&(*self.state), start, end),
                TensorInputIdent::Rhs => MA::read_window_rhs(&(*self.state), start, end),
            }
        }
    }

    /// Read the tensor at the given coordinate.
    pub fn read(&self, coordinate: u32) -> Line<EG> {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::read_lhs(&(*self.state), coordinate),
                TensorInputIdent::Rhs => MA::read_rhs(&(*self.state), coordinate),
            }
        }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::shape_lhs(&(*self.state), axis),
                TensorInputIdent::Rhs => MA::shape_rhs(&(*self.state), axis),
            }
        }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, axis: u32) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::stride_lhs(&(*self.state), axis),
                TensorInputIdent::Rhs => MA::stride_rhs(&(*self.state), axis),
            }
        }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe {
            match comptime![&self.ident] {
                TensorInputIdent::Lhs => MA::rank_lhs(&(*self.state)),
                TensorInputIdent::Rhs => MA::rank_rhs(&(*self.state)),
            }
        }
    }
}

#[cube]
impl<EG: Numeric, GA: MatmulArgs> TensorOutput<EG, GA> {
    /// Create a [tensor output](TensorOutput) from the state.
    pub fn new(state: &mut GA::State<EG>) -> TensorOutput<EG, GA> {
        TensorOutput::<EG, GA> { state }
    }

    /// Write the value to tensor at the given coordinate.
    pub fn write(&self, coordinate: u32, value: Line<EG>) {
        unsafe { GA::write_out(&mut (*self.state), coordinate, value) }
    }

    /// Get the shape of the tensor at the given axis.
    pub fn shape(&self, axis: u32) -> u32 {
        unsafe { GA::shape_out(&(*self.state), axis) }
    }

    /// Get the stride of the tensor at the given axis.
    pub fn stride(&self, dim: u32) -> u32 {
        unsafe { GA::stride_out(&(*self.state), dim) }
    }

    /// Get the rank of the tensor.
    pub fn rank(&self) -> u32 {
        unsafe { GA::rank_out(&(*self.state)) }
    }
}

#[derive(Clone)]
/// Type implementing [MatmulArgs] where all inputs and the output are materialized tensors.
///
/// Other types might implement [MatmulArgs] for fused matrix multiplication kernels.
pub struct TensorArgs;

#[derive(CubeLaunch)]
/// Input representation for [TensorArgs] implementing [MatmulArgs].
pub struct TensorInputs<EG: Numeric> {
    /// The lhs tensor.
    pub lhs: Tensor<Line<EG>>,
    /// The rhs tensor.
    pub rhs: Tensor<Line<EG>>,
}

#[cube]
impl MatmulArgs for TensorArgs {
    type Output<EG: Numeric> = Tensor<Line<EG>>;
    type Input<EG: Numeric> = TensorInputs<EG>;
    type State<EG: Numeric> = (
        *const Tensor<Line<EG>>,
        *const Tensor<Line<EG>>,
        *mut Tensor<Line<EG>>,
    );

    fn init_state<EG: Numeric>(
        input: &Self::Input<EG>,
        output: &mut Self::Output<EG>,
    ) -> Self::State<EG> {
        (&input.lhs, &input.rhs, output)
    }

    fn read_lhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG> {
        unsafe { (*state.0)[coordinate] }
    }

    fn read_rhs<EG: Numeric>(state: &Self::State<EG>, coordinate: u32) -> Line<EG> {
        unsafe { (*state.1)[coordinate] }
    }

    fn read_window_lhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>> {
        unsafe { (*state.0).slice(start, end) }
    }

    /// Read the line of the rhs tensor using the state at the given coordinate.
    fn read_window_rhs<EG: Numeric>(
        state: &Self::State<EG>,
        start: u32,
        end: u32,
    ) -> Slice<Line<EG>> {
        unsafe { (*state.1).slice(start, end) }
    }

    fn shape_lhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_rhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn shape_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.2).shape(dim) }
    }

    fn stride_lhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_rhs<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }

    fn stride_out<EG: Numeric>(state: &Self::State<EG>, dim: u32) -> u32 {
        unsafe { (*state.2).stride(dim) }
    }

    fn write_out<EG: Numeric>(state: &mut Self::State<EG>, coordinate: u32, value: Line<EG>) {
        unsafe { (*state.2)[coordinate] = value }
    }

    fn rank_lhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_rhs<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn rank_out<EG: Numeric>(state: &Self::State<EG>) -> u32 {
        unsafe { (*state.2).rank() }
    }

    fn quantization<EG: Numeric>(state: &Self::State<EG>) -> Quantization {
        let (scaling_lhs, zero_offset_lhs) = unsafe { read_quantization_params(&(*state.0)) };
        let (scaling_rhs, zero_offset_rhs) = unsafe { read_quantization_params(&(*state.1)) };
        let (scaling_out, zero_offset_out) = unsafe { read_quantization_params(&(*state.2)) };

        let scaling = scaling_lhs * scaling_rhs / scaling_out;
        let (scaling_multiplier, scaling_rounding, scaling_shift) = decompose_scaling(scaling);

        Quantization {
            zero_offset_lhs,
            zero_offset_rhs,
            zero_offset_out,
            scaling_multiplier,
            scaling_rounding,
            scaling_shift,
        }
    }
}

#[cube]
fn read_quantization_params<EG: Numeric>(tensor: &Tensor<Line<EG>>) -> (f32, i32) {
    let len = tensor.len();
    if comptime!(tensor.line_size() == 1) {
        let scaling = f32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 8][0]),
            u8::cast_from(tensor[len - 7][0]),
            u8::cast_from(tensor[len - 6][0]),
            u8::cast_from(tensor[len - 5][0]),
        ));
        let zero_offset = i32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 4][0]),
            u8::cast_from(tensor[len - 3][0]),
            u8::cast_from(tensor[len - 2][0]),
            u8::cast_from(tensor[len - 1][0]),
        ));
        (scaling, zero_offset)
    } else if comptime!(tensor.line_size() == 2) {
        let scaling = f32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 4][0]),
            u8::cast_from(tensor[len - 4][1]),
            u8::cast_from(tensor[len - 3][0]),
            u8::cast_from(tensor[len - 3][1]),
        ));
        let zero_offset = i32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 2][0]),
            u8::cast_from(tensor[len - 2][1]),
            u8::cast_from(tensor[len - 1][0]),
            u8::cast_from(tensor[len - 1][1]),
        ));
        (scaling, zero_offset)
    } else if comptime!(tensor.line_size() == 4) {
        let scaling = f32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 2][0]),
            u8::cast_from(tensor[len - 2][1]),
            u8::cast_from(tensor[len - 2][2]),
            u8::cast_from(tensor[len - 2][3]),
        ));
        let zero_offset = i32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 1][0]),
            u8::cast_from(tensor[len - 1][1]),
            u8::cast_from(tensor[len - 1][2]),
            u8::cast_from(tensor[len - 1][3]),
        ));
        (scaling, zero_offset)
    } else if comptime!(tensor.line_size() == 8) {
        let scaling = f32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 1][0]),
            u8::cast_from(tensor[len - 1][1]),
            u8::cast_from(tensor[len - 1][2]),
            u8::cast_from(tensor[len - 1][3]),
        ));
        let zero_offset = i32::bitcast_from(merge_bytes(
            u8::cast_from(tensor[len - 1][4]),
            u8::cast_from(tensor[len - 1][5]),
            u8::cast_from(tensor[len - 1][6]),
            u8::cast_from(tensor[len - 1][7]),
        ));
        (scaling, zero_offset)
    } else {
        (0.0_f32, 0) // unreachable, but I need a return value.
    }
}

#[cube]
fn merge_bytes(a: u8, b: u8, c: u8, d: u8) -> u32 {
    let mut bits = u32::cast_from(d);
    bits ^= u32::cast_from(c) << 4;
    bits ^= u32::cast_from(b) << 8;
    bits ^= u32::cast_from(a) << 12;
    bits
}

#[cube]
fn decompose_scaling(scaling: f32) -> (i64, i64, i64) {
    let log: f32 = Log2::log2(scaling);
    let log: f32 = Ceil::ceil(log);
    let log = i64::cast_from(log);

    let multiplier = i64::cast_from(f32::round(
        scaling * f32::cast_from(2_i64 << (31_i64 - log)),
    ));
    let rounding: i64 = 1_i64 << (30_i64 - log);
    let shift = 31_i64 - log;

    (multiplier, rounding, shift)
}

// impl ApproxScaling {

//     fn scale(&self, x: i32) -> i32 {
//         if self.multiplier == i32::MIN as i64 && x == i32::MIN {
//             return i32::MAX; // sature on overflow. (while multiplier is in the range of an i32 even if it is a i64)
//         }
//         let prod = (x as i64) * self.multiplier;
//         let prod_with_rounding = prod + self.rounding;
//         (prod_with_rounding >> self.shift) as i32
//     }
// }

mod __input {
    use super::*;

    impl<EG: Numeric, GA: MatmulArgs> CubeType for TensorInput<EG, GA> {
        type ExpandType = TensorInputExpand<EG, GA>;
    }

    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorInputExpand<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                ident: self.ident,
            }
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Init for TensorInputExpand<EG, GA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }
    impl<EG: Numeric, GA: MatmulArgs> CubeDebug for TensorInputExpand<EG, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }
    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorInput<EG, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<EG: Numeric, GA: MatmulArgs> Copy for TensorInput<EG, GA> {}

    impl<EG: Numeric, GA: MatmulArgs> IntoRuntime for TensorInput<EG, GA> {
        fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
            panic!("Can't exist at compile time");
        }
    }
}

mod __output {
    use super::*;

    impl<EG: Numeric, GA: MatmulArgs> CubeType for TensorOutput<EG, GA> {
        type ExpandType = TensorOutputExpand<EG, GA>;
    }

    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorOutput<EG, GA> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Clone for TensorOutputExpand<EG, GA> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
            }
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Init for TensorOutputExpand<EG, GA> {
        fn init(mut self, scope: &mut Scope) -> Self {
            self.state = self.state.init(scope);
            self
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> CubeDebug for TensorOutputExpand<EG, GA> {
        fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
            self.state.set_debug_name(scope, name);
        }
    }

    impl<EG: Numeric, GA: MatmulArgs> Copy for TensorOutput<EG, GA> {}

    impl<EG: Numeric, GA: MatmulArgs> IntoRuntime for TensorOutput<EG, GA> {
        fn __expand_runtime_method(self, _scope: &mut Scope) -> Self::ExpandType {
            panic!("Can't exist at compile time");
        }
    }
}
