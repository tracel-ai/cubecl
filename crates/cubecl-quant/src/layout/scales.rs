use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};
use cubecl_std::{
    FastDivmod, FastDivmodArgs,
    tensor::{
        launch::{TypedView, TypedViewLaunch},
        layout::{Coords1d, Layout, LayoutExpand},
    },
};

use crate::scheme::{QuantLevel, QuantScheme};

/// Layout for quantization scales, indexed by quant element index and returns the corresponding
/// scale based on the quantization type.
#[derive(CubeType, CubeLaunch)]
pub enum ScalesLayout {
    PerTensor(PerTensorLayout),
    BlockScaled(BlockScaledLayout),
}

#[cube]
impl Layout for ScalesLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        match self {
            ScalesLayout::PerTensor(layout) => layout.to_source_pos(pos),
            ScalesLayout::BlockScaled(layout) => layout.to_source_pos(pos),
        }
    }

    fn shape(&self) -> Self::Coordinates {
        match self {
            ScalesLayout::PerTensor(layout) => layout.shape(),
            ScalesLayout::BlockScaled(layout) => layout.shape(),
        }
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        match self {
            ScalesLayout::PerTensor(layout) => layout.is_in_bounds(pos),
            ScalesLayout::BlockScaled(layout) => layout.is_in_bounds(pos),
        }
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        match self {
            ScalesLayout::PerTensor(layout) => layout.to_source_pos_checked(pos),
            ScalesLayout::BlockScaled(layout) => layout.to_source_pos_checked(pos),
        }
    }
}

#[cube]
impl ScalesLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: u32) -> bool {
        match self {
            ScalesLayout::PerTensor(layout) => layout.is_block_start(pos),
            ScalesLayout::BlockScaled(layout) => layout.is_block_start(pos),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct PerTensorLayout {
    tensor_len: u32,
}

#[cube]
impl Layout for PerTensorLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, _pos: Self::Coordinates) -> Self::SourceCoordinates {
        0u32.runtime()
    }

    fn shape(&self) -> Self::Coordinates {
        self.tensor_len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.tensor_len
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

#[cube]
impl PerTensorLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: u32) -> bool {
        pos == 0
    }
}

#[derive(CubeType, CubeLaunch)]
pub struct BlockScaledLayout {
    tensor_shape: Sequence<FastDivmod>,
    tensor_len: u32,
    scales_strides: Sequence<u32>,
    #[cube(comptime)]
    block_size: Vec<u8>,
    #[cube(comptime)]
    scales_line_size: u32,
}

#[cube]
impl Layout for BlockScaledLayout {
    type Coordinates = Coords1d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let rank = comptime![self.scales_strides.len()];
        let mut offs = pos;
        let mut scale_offs = 0;

        #[unroll]
        for i in 0..rank {
            let i = unwrap(i);
            let dim = comptime![rank - i - 1];
            let block_size_local = comptime![self.block_size[dim as usize] as u32];
            let (rem, offs_local) = self.tensor_shape.index(dim).div_mod(offs);
            offs = rem;
            scale_offs += (offs_local / block_size_local) * *self.scales_strides.index(dim);
        }

        scale_offs / self.scales_line_size
    }

    fn shape(&self) -> Self::Coordinates {
        self.tensor_len
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        pos < self.tensor_len
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }
}

#[cube]
impl BlockScaledLayout {
    /// Whether the position is at the start of a new block. Used for electing a unit to write each
    /// scale.
    pub fn is_block_start(&self, pos: u32) -> bool {
        let rank = comptime![self.scales_strides.len()];
        let mut offs = pos;
        let mut is_start = true;

        #[unroll]
        for i in 0..rank {
            let i = unwrap(i);
            let dim = comptime![rank - i - 1];
            let block_size_local = comptime![self.block_size[dim as usize] as u32];
            let (rem, offs_local) = self.tensor_shape.index(dim).div_mod(offs);
            offs = rem;
            is_start &= offs_local.is_multiple_of(block_size_local);
        }

        is_start
    }
}

#[allow(unused_variables)]
#[cube]
fn unwrap(v: u32) -> comptime_type!(u32) {
    intrinsic!(|_| v.constant().expect("Must be constant").as_u32())
}

/// [TensorView] with a linear layout inferred from the shape/strides at launch.
/// Useful for elementwise kernels.
pub type ScalesView<E, IO = ReadOnly> = TypedView<E, ScalesLayout, IO>;
/// Launch type for [LinearTensorView].
pub type ScalesViewLaunch<'a, R> = TypedViewLaunch<'a, ScalesLayout, R>;

/// Create a scales view from the values and scales handle, line size and quantization scheme.
/// `values` should be *the quantized tensor*, and will be adjusted by `num_quants`.
pub fn scales_view<'a, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    values: &'a TensorHandleRef<'a, R>,
    scales: &'a TensorHandleRef<'a, R>,
    scales_line_size: u8,
    quant_scheme: &QuantScheme,
) -> ScalesViewLaunch<'a, R> {
    let layout = scales_layout(client, values, scales, scales_line_size, quant_scheme);
    let len = scales.shape.iter().product::<usize>();
    let buffer = unsafe {
        ArrayArg::from_raw_parts_and_size(scales.handle, len, scales_line_size, scales.elem_size)
    };
    ScalesViewLaunch::new(buffer, layout)
}

pub fn scales_layout<'a, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    values: &'a TensorHandleRef<'a, R>,
    scales: &'a TensorHandleRef<'a, R>,
    scales_line_size: u8,
    scheme: &QuantScheme,
) -> ScalesLayoutArgs<'a, R> {
    let values_len = values.shape.iter().product::<usize>() * scheme.num_quants();
    let values_len = ScalarArg::new(values_len as u32);

    match &scheme.level {
        QuantLevel::Tensor => ScalesLayoutArgs::PerTensor(PerTensorLayoutLaunch::new(values_len)),
        QuantLevel::Block(block_size) => {
            let tensor_shape = shape_divmod_quant(client, values.shape, scheme.num_quants());
            let scales_strides = strides_seq(scales.strides);
            ScalesLayoutArgs::BlockScaled(BlockScaledLayoutLaunch::new(
                tensor_shape,
                values_len,
                scales_strides,
                block_size.to_dim_vec(values.shape.len()),
                scales_line_size as u32,
            ))
        }
    }
}

fn shape_divmod_quant<'a, R: Runtime>(
    client: &ComputeClient<R::Server, R::Channel>,
    shape: &'a [usize],
    num_quants: usize,
) -> SequenceArg<'a, R, FastDivmod> {
    let mut out_seq = SequenceArg::new();
    for s in &shape[..shape.len() - 1] {
        out_seq.push(FastDivmodArgs::new(client, *s as u32));
    }
    let last = *shape.last().unwrap() * num_quants;
    out_seq.push(FastDivmodArgs::new(client, last as u32));
    out_seq
}

fn strides_seq<'a, R: Runtime>(strides: &'a [usize]) -> SequenceArg<'a, R, u32> {
    let mut out_seq = SequenceArg::new();
    for s in strides {
        out_seq.push(ScalarArg::new(*s as u32));
    }
    out_seq
}
