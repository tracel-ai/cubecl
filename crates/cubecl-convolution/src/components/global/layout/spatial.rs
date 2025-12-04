use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_std::tensor::{
    layout::{
        Coordinates, Coords1d, Layout, LayoutExpand,
        as_dyn::{IntoDyn, IntoDynExpand},
    },
    r#virtual::VirtualTensor,
};

use crate::components::Dimensionality;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct NhwcCoords {
    pub batch: u32,
    pub spatial: Sequence<i32>,
    pub channel: u32,
}

#[cube]
impl IntoDyn for NhwcCoords {
    fn into_dyn(self) -> Sequence<i32> {
        let mut seq = Sequence::new();
        seq.push(self.batch as i32);
        for x in self.spatial {
            seq.push(x);
        }
        seq.push(self.channel as i32);
        seq
    }
}

type NhwcTuple = (u32, Sequence<i32>, u32);

#[cube]
impl NhwcCoords {
    pub fn new(batch: u32, spatial: Sequence<i32>, channel: u32) -> Self {
        NhwcCoords {
            batch,
            spatial,
            channel,
        }
    }

    fn into_tuple(self) -> NhwcTuple {
        (self.batch, self.spatial, self.channel)
    }

    fn from_tuple(tuple: NhwcTuple) -> Self {
        NhwcCoords::new(tuple.0, tuple.1, tuple.2)
    }
}

#[cube]
impl Coordinates for NhwcCoords {
    fn add(this: Self, other: Self) -> Self {
        let tuple = NhwcTuple::add(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn sub(this: Self, other: Self) -> Self {
        let tuple = NhwcTuple::sub(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn min(this: Self, other: Self) -> Self {
        let tuple = <NhwcTuple as Coordinates>::min(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn max(this: Self, other: Self) -> Self {
        let tuple = <NhwcTuple as Coordinates>::max(this.into_tuple(), other.into_tuple());
        NhwcCoords::from_tuple(tuple)
    }

    fn is_in_bounds(pos: &Self, bounds: &Self) -> bool {
        NhwcTuple::is_in_bounds(&pos.clone().into_tuple(), &bounds.clone().into_tuple())
    }

    fn from_int(this: &Self, #[comptime] value: i64) -> Self {
        let tuple = NhwcTuple::from_int(&this.clone().into_tuple(), value);
        NhwcCoords::from_tuple(tuple)
    }
}

/// Layout for a spatial (i.e. NHWC) tensor. Bounds check only applies to spatial dimensions, not
/// channel or batch (because these are implicitly checked in the layouts used with spatial tensors).
#[derive(CubeType, CubeLaunch, Clone)]
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
    #[cube(comptime)]
    pub check_channel: bool,
}

#[cube]
impl NhwcLayout {
    pub fn new<E: Numeric, IO: Clone>(
        tensor: VirtualTensor<E, IO>,
        #[comptime] dim: Dimensionality,
        #[comptime] check_spatial: bool,
        #[comptime] check_channel: bool,
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
            check_channel,
        }
    }
}

#[cube]
impl Layout for NhwcLayout {
    type Coordinates = NhwcCoords;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let NhwcCoords {
            batch,
            spatial,
            channel,
        } = pos;

        let spatial_dims = self.shapes_spatial.len();
        let mut read_pos = batch * self.stride_batch + channel * self.stride_channel;

        #[unroll]
        for i in 0..spatial_dims {
            read_pos += *spatial.index(i) as u32 * *self.strides_spatial.index(i);
        }

        read_pos / self.line_size
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos.clone()), self.is_in_bounds(pos))
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut in_bounds = true.runtime();
        if comptime![self.check_spatial] {
            let spatial_dims = self.shapes_spatial.len();

            #[unroll]
            for i in 0..spatial_dims {
                let pos = *pos.spatial.index(i);
                in_bounds &= pos >= 0 && (pos as u32) < *self.shapes_spatial.index(i);
            }
        }
        if comptime![self.check_channel] {
            in_bounds &= pos.channel < self.shape_channel;
        }

        in_bounds
    }

    fn shape(&self) -> Self::Coordinates {
        NhwcCoords {
            batch: self.shape_batch,
            spatial: cast_seq(self.shapes_spatial.clone()),
            channel: self.shape_channel,
        }
    }
}

#[cube]
pub(crate) fn cast_seq<From: CubePrimitive, To: CubePrimitive>(
    seq: Sequence<From>,
) -> Sequence<To> {
    let num_elems = seq.len();
    let mut out_seq = Sequence::new();
    #[unroll]
    for i in 0..num_elems {
        let elem = To::cast_from(*seq.index(i));
        out_seq.push(elem);
    }
    out_seq
}

impl<'a, R: Runtime> NhwcLayoutLaunch<'a, R> {
    pub fn from_handle(
        handle: &TensorHandleRef<'a, R>,
        line_size: u32,
        check_spatial: bool,
        check_channel: bool,
    ) -> Self {
        let rank = handle.shape.len();
        let dim_c = rank - 1;

        let stride_batch = ScalarArg::new(handle.strides[0] as u32);
        let strides_spatial = handle.strides[1..dim_c]
            .iter()
            .map(|s| ScalarArg::new(*s as u32))
            .collect();
        let stride_channel = ScalarArg::new(handle.strides[dim_c] as u32);

        let shape_batch = ScalarArg::new(handle.shape[0] as u32);
        let shapes_spatial = handle.shape[1..dim_c]
            .iter()
            .map(|s| ScalarArg::new(*s as u32))
            .collect();
        let shape_channel = ScalarArg::new(handle.shape[dim_c] as u32);

        Self::new(
            stride_batch,
            strides_spatial,
            stride_channel,
            shape_batch,
            shapes_spatial,
            shape_channel,
            line_size,
            check_spatial,
            check_channel,
        )
    }
}
