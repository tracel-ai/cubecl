use cubecl::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};
use cubecl_std::tensor::{
    layout::{
        Coordinates, Coords1d, Layout, VirtualLayoutOperations, VirtualLayoutOperationsExpand,
    },
    r#virtual::VirtualTensor,
};

use crate::components::{Dimensionality, global::layout::virtual_layout};

#[derive(CubeType, Clone)]
pub struct NhwcCoords {
    pub batch: u32,
    pub spatial: Sequence<i32>,
    pub channel: u32,
}

impl NhwcCoordsExpand {
    pub fn __expand_clone_method(&self, _scope: &mut Scope) -> Self {
        NhwcCoordsExpand {
            batch: self.batch.clone(),
            spatial: self.spatial.clone(),
            channel: self.channel.clone(),
        }
    }
}

impl Coordinates for NhwcCoords {}

/// Layout for a spatial (i.e. NHWC) tensor. Bounds check only applies to spatial dimensions, not
/// channel or batch (because these are implicitly checked in the layouts used with spatial tensors).
#[derive(CubeType, Clone)]
pub struct NhwcLayout {
    /// Stride for N
    pub stride_batch: u32,
    /// Strides for DHW
    pub strides_spatial: Sequence<u32>,
    /// Stride for C
    pub stride_channel: u32,

    /// Shape of N
    pub shape_batch: u32,
    /// Shape of DHW
    pub shapes_spatial: Sequence<u32>,
    /// Shape of C
    pub shape_channel: u32,

    #[cube(comptime)]
    pub line_size: u32,
    #[cube(comptime)]
    pub check_spatial: bool,
}

#[cube]
impl NhwcLayout {
    pub fn new<E: Numeric, IO: Clone>(
        tensor: VirtualTensor<E, IO>,
        #[comptime] dim: Dimensionality,
        #[comptime] check_spatial: bool,
    ) -> Self {
        let spatial_dims = comptime![dim.num_dims()];
        let mut strides_spatial = Sequence::new();
        let mut shapes_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
            shapes_spatial.push(tensor.shape(i + 1));
        }

        let stride_batch = tensor.stride(0);
        let stride_channel = tensor.stride(spatial_dims + 1);

        let shape_batch = tensor.shape(0);
        let shape_channel = tensor.shape(spatial_dims + 1);

        NhwcLayout {
            stride_batch,
            strides_spatial,
            stride_channel,
            shape_batch,
            shapes_spatial,
            shape_channel,
            line_size: tensor.line_size(),
            check_spatial,
        }
    }
}

#[cube]
impl Layout for NhwcLayout {
    type Coordinates = NhwcCoords;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(this: &Self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let NhwcCoords {
            batch,
            spatial,
            channel,
        } = pos;

        let spatial_dims = this.shapes_spatial.len();
        let mut read_pos = batch * this.stride_batch + channel * this.stride_channel;

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            read_pos += *spatial.index(i) as u32 * *this.strides_spatial.index(i);
        }

        read_pos / this.line_size
    }

    fn to_source_pos_checked(
        this: &Self,
        pos: Self::Coordinates,
    ) -> (Self::SourceCoordinates, bool) {
        (this.to_source_pos(pos.clone()), this.is_in_bounds(pos))
    }

    fn is_in_bounds(this: &Self, pos: Self::Coordinates) -> bool {
        if comptime![this.check_spatial] {
            let spatial_dims = this.shapes_spatial.len();
            let mut spatial_in_bounds = true;

            #[unroll]
            for i in 0..spatial_dims {
                let i = unwrap(i);
                let pos = *pos.spatial.index(i);
                spatial_in_bounds &= pos >= 0 && (pos as u32) < *this.shapes_spatial.index(i);
            }

            spatial_in_bounds
        } else {
            true.runtime()
        }
    }

    fn shape(this: &Self) -> Self::Coordinates {
        NhwcCoords {
            batch: this.shape_batch,
            spatial: cast_seq(this.shapes_spatial.clone()),
            channel: this.shape_channel,
        }
    }
}

virtual_layout!(NhwcLayout, NhwcLayoutExpand);

#[allow(unused_variables)]
#[cube]
pub(crate) fn unwrap(v: u32) -> comptime_type!(u32) {
    intrinsic!(|_| v.constant().expect("Must be constant").as_u32())
}

#[cube]
pub(crate) fn cast_seq<From: CubePrimitive, To: CubePrimitive>(
    seq: Sequence<From>,
) -> Sequence<To> {
    let num_elems = seq.len();
    let mut out_seq = Sequence::new();
    #[unroll]
    for i in 0..num_elems {
        let i = unwrap(i);
        let elem = To::cast_from(*seq.index(i));
        out_seq.push(elem);
    }
    out_seq
}
