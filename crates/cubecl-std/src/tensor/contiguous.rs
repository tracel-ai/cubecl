use crate::{
    FastDivmod, FastDivmodArgs,
    tensor::layout::{Coords1d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand},
};

use super::TensorHandle;
use cubecl::prelude::*;
use cubecl_core::{self as cubecl, calculate_cube_count_elemwise, tensor_line_size_parallel};

pub const NUM_SM_APPROX: u32 = 50;

/// Returns the offset of the tensor corresponding to the layout tensor.
#[cube]
pub fn index_offset_with_layout<N: CubePrimitive, L: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    layout: &Tensor<Line<L>>,
    offset_layout: u32,
    dim_start: u32,
    dim_end: u32,
    #[comptime] unroll: bool,
) -> u32 {
    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;

    #[unroll(unroll)]
    for i in dim_start..dim_end {
        let ogwl = offset_ref / layout.stride(i);
        offset += ogwl % tensor.shape(i) * tensor.stride(i);
    }

    offset / tensor.line_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous<N: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    offset_layout: u32,
    #[comptime] rank: Option<u32>,
) -> u32 {
    let unroll = rank.is_some();
    let rank = rank.unwrap_or_else(|| tensor.rank());

    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;
    let mut remainder = offset_ref;

    #[unroll(unroll)]
    for i in 0..rank {
        let dim = rank - i - 1;
        let shape = tensor.shape(dim);
        let ogwl = remainder % shape;
        offset += ogwl * tensor.stride(dim);
        remainder /= shape;
    }

    offset / tensor.line_size()
}

/// Returns the offset of the tensor corresponding to a contiguous layout.
#[cube]
pub fn index_offset_contiguous_fastdivmod<N: CubePrimitive>(
    tensor: &Tensor<Line<N>>,
    offset_layout: u32,
    shape: &Sequence<FastDivmod>,
    stride: &Sequence<u32>,
) -> u32 {
    let rank = comptime![shape.len()];

    let offset_ref = offset_layout * tensor.line_size();
    let mut offset = 0;
    let mut remainder = offset_ref;

    let mut dim = comptime![rank - 1];

    #[unroll]
    for _ in 0..rank {
        let shape = shape.index(dim);
        let (rem, ogwl) = shape.div_mod(remainder);
        offset += ogwl * stride.index(dim);
        remainder = rem;

        comptime![dim = dim.saturating_sub(1);]
    }

    offset / tensor.line_size()
}

/// Layout for tensor that may or may not be strided on the last dimension. Efficiently translates
/// the absolute index to strided index.
#[derive(CubeType, CubeLaunch, Clone)]
pub enum StridedLayoutType {
    Pitched { shape: FastDivmod, stride: u32 },
    None,
}

#[derive(CubeType, CubeLaunch, Clone)]
pub struct StridedLayout {
    ty: StridedLayoutType,
    len: u32,
    #[cube(comptime)]
    line_size: u8,
}

impl<'a, R: Runtime> StridedLayoutLaunch<'a, R> {
    pub fn from_shape_strides(
        client: &ComputeClient<R::Server, R::Channel>,
        shape: &[usize],
        strides: &[usize],
        line_size: &'a u8,
    ) -> Self {
        let rank = shape.len();
        let len = shape.iter().product::<usize>();
        let ty = if rank == 1 || is_contiguous(shape, strides) {
            StridedLayoutTypeArgs::None
        } else {
            StridedLayoutTypeArgs::Pitched {
                shape: FastDivmodArgs::new(client, shape[rank - 1] as u32),
                stride: ScalarArg::new(strides[rank - 2] as u32),
            }
        };
        Self::new(ty, ScalarArg::new(len as u32), line_size)
    }

    pub fn from_handle(
        client: &ComputeClient<R::Server, R::Channel>,
        handle: &TensorHandleRef<'_, R>,
        line_size: &'a u8,
    ) -> Self {
        Self::from_shape_strides(client, handle.shape, handle.strides, line_size)
    }
}

#[cube]
impl Layout for StridedLayout {
    type Coordinates = Coords1d;

    fn to_linear_pos(this: &Self, pos: Self::Coordinates) -> u32 {
        match &this.ty {
            StridedLayoutType::Pitched { shape, stride } => {
                let offset_abs = pos * comptime![this.line_size as u32];
                let (y, x) = shape.div_mod(offset_abs);
                let offset = y * stride + x;
                offset / comptime![this.line_size as u32]
            }
            StridedLayoutType::None => pos,
        }
    }

    fn to_linear_pos_checked(this: &Self, pos: Self::Coordinates) -> (u32, bool) {
        let idx = this.to_linear_pos(pos);
        let in_bounds = pos < this.len;
        (idx, in_bounds)
    }

    fn shape(this: &Self) -> Self::Coordinates {
        this.len
    }
}

mod r#virtual {
    use crate::tensor::layout::{VirtualLayout, VirtualLayoutOperationsExpand};

    use super::*;

    impl VirtualLayoutOperationsExpand<Coords1d> for StridedLayoutExpand {
        fn __expand_to_linear_pos_method(
            &self,
            scope: &mut Scope,
            pos: ExpandElementTyped<u32>,
        ) -> <u32 as CubeType>::ExpandType {
            StridedLayout::__expand_to_linear_pos(scope, self.clone(), pos)
        }
        fn __expand_to_linear_pos_checked_method(
            &self,
            scope: &mut cubecl_core::prelude::Scope,
            pos: ExpandElementTyped<u32>,
        ) -> <(u32, bool) as cubecl_core::prelude::CubeType>::ExpandType {
            StridedLayout::__expand_to_linear_pos_checked(scope, self.clone(), pos)
        }
        fn __expand_shape_method(
            &self,
            scope: &mut cubecl_core::prelude::Scope,
        ) -> ExpandElementTyped<u32> {
            StridedLayout::__expand_shape(scope, self.clone())
        }
    }
    #[cube]
    impl StridedLayout {
        pub fn virt(self) -> VirtualLayout<Coords1d> {
            VirtualLayout::new::<StridedLayout>(self)
        }
    }
}

#[cube(launch)]
fn into_contiguous_kernel<N: CubePrimitive>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: StridedLayout,
    shape: Sequence<FastDivmod>,
    stride: Sequence<u32>,
    #[comptime] elems_per_thread: u32,
) {
    let offset_output = ABSOLUTE_POS * elems_per_thread;
    let line_size = input.line_size();

    let mut registers = Array::<Line<N>>::vectorized(elems_per_thread, line_size);

    #[unroll]
    for i in 0..elems_per_thread {
        let offset_input =
            index_offset_contiguous_fastdivmod::<N>(input, offset_output + i, &shape, &stride);
        registers[i] = input[offset_input];
    }

    let offset_output = out_layout.to_linear_pos(offset_output);

    #[unroll]
    for i in 0..elems_per_thread {
        output[offset_output + i] = registers[i];
    }
}

#[cube(launch)]
fn into_contiguous_kernel_pack<N: CubePrimitive>(
    input: &Tensor<Line<N>>,
    output: &mut Tensor<Line<N>>,
    out_layout: StridedLayout,
    shape: Sequence<FastDivmod>,
    stride: Sequence<u32>,
    #[comptime] elems_per_thread: u32,
) {
    let line_size = output.line_size();
    let lines_per_thread = comptime![elems_per_thread / line_size];

    let offset_output = ABSOLUTE_POS * lines_per_thread;
    let offset_input = offset_output * line_size;

    let mut registers = Array::<Line<N>>::vectorized(lines_per_thread, line_size);

    #[unroll]
    for i in 0..lines_per_thread {
        let offset = i * line_size;
        let mut reg = Line::<N>::empty(line_size);
        #[unroll]
        for k in 0..line_size {
            let offset_input = offset_input + offset + k;
            let offset_input =
                index_offset_contiguous_fastdivmod::<N>(input, offset_input, &shape, &stride);
            reg[k] = input[offset_input][0];
        }
        registers[i] = reg;
    }

    let offset_output = out_layout.to_linear_pos(offset_output);

    #[unroll]
    for i in 0..lines_per_thread {
        output[offset_output + i] = registers[i];
    }
}

/// Make a jit tensor contiguous.
pub fn into_contiguous<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
) -> TensorHandle<R, E> {
    let num_elems: usize = input.shape.iter().product();

    let handle = client.empty(num_elems * size_of::<E>());
    let output = TensorHandle::new_contiguous(input.shape.to_vec(), handle);

    into_contiguous_ref::<R, E>(client, input, &output.as_ref());

    output
}

/// Make a jit tensor contiguous, using the pitched allocator if available.
/// See [create_tensor](cubecl_runtime::client::ComputeClient::create_tensor).
pub fn into_contiguous_pitched<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
) -> TensorHandle<R, E> {
    if input.shape.len() <= 1 {
        return into_contiguous(client, input);
    }

    let output = TensorHandle::empty(client, input.shape.to_vec());

    into_contiguous_ref::<R, E>(client, input, &output.as_ref());

    output
}

/// Make a jit tensor contiguous.
pub fn into_contiguous_ref<R: Runtime, E: CubePrimitive>(
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<'_, R>,
    output: &TensorHandleRef<'_, R>,
) {
    let num_elems: usize = input.shape.iter().product();

    // Vectorization is only enabled when the last dimension is contiguous.
    let rank = input.strides.len();
    let vectorization_factor = tensor_line_size_parallel(
        R::supported_line_sizes().iter().cloned(),
        input.shape,
        input.strides,
        rank - 1,
    );
    let num_vecs = num_elems / vectorization_factor as usize;
    let num_sm = client
        .properties()
        .hardware
        .num_streaming_multiprocessors
        .unwrap_or(NUM_SM_APPROX);
    let simul_vecs = num_sm * CubeDim::default().num_elems();
    let mut elems_per_unit = match num_vecs as u32 / simul_vecs {
        0..2 => 1,
        2..4 => 2,
        4..8 => 4,
        8.. => 8,
    };

    let mut num_elems_per_unit = vectorization_factor as u32 * elems_per_unit;

    let last_dim = output.shape[rank - 1];
    let is_padded = rank > 1 && last_dim != output.strides[rank - 2];

    // If tensor is strided, elems_per_unit must be compatible with last dim
    while is_padded && last_dim % num_elems_per_unit as usize != 0 {
        elems_per_unit /= 2;
        num_elems_per_unit /= 2;
    }

    let out_vec = if vectorization_factor > 1 {
        vectorization_factor
    } else {
        *R::supported_line_sizes()
            .iter()
            .filter(|it| num_elems_per_unit % **it as u32 == 0)
            .max()
            .unwrap_or(&1)
    };

    let out_layout = StridedLayoutLaunch::from_handle(client, output, &out_vec);

    let cube_dim = CubeDim::default();
    let cube_count =
        calculate_cube_count_elemwise(num_elems.div_ceil(num_elems_per_unit as usize), cube_dim);

    let shape = SequenceArg {
        values: input
            .shape
            .iter()
            .map(|dim| FastDivmodArgs::new(client, *dim as u32))
            .collect(),
    };

    let stride = SequenceArg {
        values: input
            .strides
            .iter()
            .map(|s| ScalarArg::new(*s as u32))
            .collect(),
    };

    let launch = if vectorization_factor != out_vec && out_vec > 1 {
        into_contiguous_kernel_pack::launch::<E, R>
    } else {
        into_contiguous_kernel::launch::<E, R>
    };

    launch(
        client,
        cube_count,
        cube_dim,
        input.as_tensor_arg(vectorization_factor),
        output.as_tensor_arg(out_vec),
        out_layout,
        shape,
        stride,
        elems_per_unit,
    );
}

/// Checks if the tensor associated with the given shape and strides is contiguous.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    for (expected, &stride) in compact_strides(shape).into_iter().zip(strides) {
        if expected != stride {
            return false;
        }
    }

    true
}

pub fn compact_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![1; rank];
    for i in (0..rank - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
