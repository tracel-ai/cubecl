use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, intrinsic};
use cubecl_matmul::components::MatmulIdent;
use cubecl_std::{FastDivmod, tensor::r#virtual::VirtualTensor};

use crate::{ConvGemmConfig, loader::im2col_tma::div_mod_seq};

#[derive(CubeType)]
/// A view of a feature map tensor that starts reading data from a specified offset.
/// Ensures safe access by preventing out-of-bounds errors.
/// Includes pre-fetched shapes and strides for optimized performance.
pub struct Im2colReader<E: Numeric> {
    pub tensor: VirtualTensor<E>,
    pub m_offset: u32,
    pub k_offset: u32,

    pub stride_batch: u32,
    pub strides_spatial: Sequence<u32>,
    pub stride_channel: u32,

    pub shapes_spatial: Sequence<u32>,
    pub shape_channel: u32,

    pub shape_out: Sequence<FastDivmod>,

    pub shape_m: u32,
    pub shape_k: u32,
}

#[cube]
impl<E: Numeric> Im2colReader<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tensor: VirtualTensor<E>,
        shape_out: Sequence<FastDivmod>,
        x_offset: u32,
        y_offset: u32,
        shape_k: u32,
        shape_m: u32,
    ) -> Im2colReader<E> {
        let spatial_dims = comptime![shape_out.len()];
        let mut strides_spatial = Sequence::new();
        let mut shapes_spatial = Sequence::new();

        #[unroll]
        for i in 0..spatial_dims {
            strides_spatial.push(tensor.stride(i + 1));
            shapes_spatial.push(tensor.shape(i + 1));
        }

        let stride_batch = tensor.stride(0);
        let stride_channel = tensor.stride(spatial_dims + 1);

        let shape_channel = tensor.shape(spatial_dims + 1);

        Im2colReader::<E> {
            tensor,
            m_offset: x_offset,
            k_offset: y_offset,
            stride_batch,
            strides_spatial,
            stride_channel,
            shapes_spatial,
            shape_channel,
            shape_out,
            shape_m,
            shape_k,
        }
    }
}

unsafe impl<E: Numeric> Sync for Im2colReader<E> {}
unsafe impl<E: Numeric> Send for Im2colReader<E> {}

#[cube]
impl<E: Numeric> Im2colReader<E> {
    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn update_view(&mut self, k_offset: u32) {
        self.k_offset += k_offset;
    }

    /// Reads data from the tensor view at the specified tile coordinates (tile_x, tile_y) using
    /// the `im2col` algorithm to translate them to input coordinates.
    ///
    /// Each unit loads one line in a coalesced manner for improved efficiency.
    /// For row-major tensors, subsequent units read lines horizontally within the tile,
    /// while for column-major tensors, they read lines vertically.
    ///
    /// # Note
    ///
    /// Out-of-bounds reads will be translated to zeros.
    pub fn load_simple<G: ConvGemmConfig>(
        &self,
        tile_x: u32,
        tile_y: u32,
        unit_id: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Line<E> {
        let line_size = config.global_line_size(ident);
        let tile_size_x = config.tiling_scheme().elements_in_tile_row(ident);
        let tile_size_y = config.tiling_scheme().elements_in_tile_col(ident);

        let view_tile_m = tile_x * tile_size_x + self.m_offset;
        let view_tile_k = tile_y * tile_size_y + self.k_offset;

        let load_m = unit_id / tile_size_y;
        let load_k = unit_id % tile_size_y;

        let view_m = view_tile_m + load_m;
        let view_k = view_tile_k + load_k;

        let (batch, out_offs) = div_mod_seq(view_m, &self.shape_out);

        let channel = view_k % self.shape_channel;
        let mut rem = view_k / self.shape_channel;

        let spatial_dims = comptime![self.shapes_spatial.len()];
        let mut in_pos = Sequence::<i32>::new();

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            let dim = comptime![spatial_dims - i - 1];
            let ksize = comptime![config.kernel_size(dim)];
            let k_pos = rem % ksize;
            rem /= ksize;

            let out_pos = *out_offs.index(dim);
            let stride = comptime![config.stride(dim)];
            let dilate = comptime![config.dilation(dim)];
            let pad = comptime![config.padding(dim)];

            let pos = (out_pos * stride + k_pos * dilate) as i32 - pad;
            in_pos.push(pos);
        }

        let in_pos = in_pos.rev();

        let has_padding = comptime! {
            let mut has_padding = false;
            for i in 0..spatial_dims {
                has_padding |= config.padding(i) != 0;
            }
            has_padding
        };

        let m_in_bounds =
            comptime!(!config.check_row_bounds(MatmulIdent::Lhs)) || view_m < self.shape_m;
        let k_in_bounds =
            comptime!(!config.check_col_bounds(MatmulIdent::Lhs)) || view_k < self.shape_k;
        let mut spatial_in_bounds = true;

        if has_padding {
            #[unroll]
            for i in 0..spatial_dims {
                let i = unwrap(i);
                let pos = *in_pos.index(i);
                spatial_in_bounds &= pos >= 0 && (pos as u32) < *self.shapes_spatial.index(i);
            }
        }

        let in_bounds = m_in_bounds && k_in_bounds && spatial_in_bounds;

        let mut read_pos = batch * self.stride_batch + channel * self.stride_channel;

        #[unroll]
        for i in 0..spatial_dims {
            let i = unwrap(i);
            read_pos += *in_pos.index(i) as u32 * *self.strides_spatial.index(i);
        }

        let read_pos = read_pos / line_size;

        let mut res = Line::empty(line_size).fill(E::from_int(0));
        if in_bounds {
            res = self.read(read_pos);
        }

        res
    }

    fn read(&self, position: u32) -> Line<E> {
        self.tensor.read(position)
    }
}

#[allow(unused_variables)]
#[cube]
fn unwrap(v: u32) -> comptime_type!(u32) {
    intrinsic!(|_| v.constant().expect("Must be constant").as_u32())
}
