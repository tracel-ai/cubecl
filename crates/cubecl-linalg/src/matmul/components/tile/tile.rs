use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::components::{Ident, InputIdent, MatrixLayout, stage::Skew};

use super::TileConfig;

#[derive(CubeType)]
/// Contiguous row (when row major) or column (when col major)
/// The segment does not own its data but rather metadata to find the data within the tile slice
pub struct Segment<ES: Numeric> {
    /// The slice of the whole tile
    tile_slice: Slice<Line<ES>>,
    /// Where the segment starts on the tile slice
    pub offset: u32,
    #[cube(comptime)]
    /// Number of data elements, without skew, lined with stage_line_size
    pub num_data: u32,
    #[cube(comptime)]
    pub skew: Skew,
}

#[cube]
impl<ES: Numeric> Segment<ES> {
    pub fn as_data_slice_mut(&mut self, #[comptime] read_line_size: u32) -> SliceMut<Line<ES>> {
        self.tile_slice
            .slice_mut(self.offset, self.offset + self.num_data)
            .with_line_size(read_line_size)
    }
}

#[derive(CubeType)]
/// Data to be handed to the tile matmul
pub struct Tile<ES: Numeric> {
    /// Slice containing all segments
    pub slice: Slice<Line<ES>>,
    #[cube(comptime)]
    /// Number of segments
    pub num_segments: u32,
    #[cube(comptime)]
    /// Number of data elements, without skew, lined with stage_line_size
    pub segment_length: u32,
    #[cube(comptime)]
    /// Stride between each segment,
    pub stride: u32,
    #[cube(comptime)]
    pub skew: Skew,
}

#[cube]
impl<ES: Numeric> Tile<ES> {
    pub fn new_contiguous<T: TileConfig>(
        tile_slice: Slice<Line<ES>>,
        #[comptime] skew: Skew,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> Tile<ES> {
        let (num_segments, segment_length) = Tile::<ES>::segment_info::<T>(ident, config);
        let stride = comptime!(segment_length + skew.padding_size());

        Tile::<ES> {
            slice: tile_slice,
            num_segments,
            segment_length,
            stride,
            skew,
        }
    }

    pub fn segment_info<T: TileConfig>(
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> comptime_type!((u32, u32)) {
        let tile_shape = config.tile_shape();
        let (num_segments, segment_length) = comptime! {match ident.as_input_ident() {
            InputIdent::Lhs => match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => (tile_shape.m, tile_shape.k / config.stage_line_size(ident)),
                MatrixLayout::ColMajor => (tile_shape.k, tile_shape.m / config.stage_line_size(ident)),
            },
            InputIdent::Rhs => match config.matrix_layout(ident) {
                MatrixLayout::RowMajor => (tile_shape.k, tile_shape.n/config.stage_line_size(ident)),
                MatrixLayout::ColMajor => (tile_shape.n, tile_shape.k/config.stage_line_size(ident)),
            },
        }};

        (num_segments, segment_length)
    }

    // pub fn make_segments(
    //     tile_slice: Slice<Line<ES>>,
    //     #[comptime] num_segments: u32,
    //     #[comptime] segment_length: u32,
    //     #[comptime] stride: u32,
    //     #[comptime] skew: Skew,
    // ) -> Sequence<Segment<ES>> {
    //     let mut segments = Sequence::new();
    //     let mut segment_iter = comptime![0];

    //     #[allow(clippy::explicit_counter_loop)]
    //     #[unroll]
    //     for _ in 0..num_segments {
    //         segments.push(Segment::<ES> {
    //             tile_slice,
    //             offset: comptime!(segment_iter * stride),
    //             num_data: segment_length,
    //             skew,
    //             _phantom: PhantomData,
    //         });

    //         comptime![segment_iter += 1];
    //     }

    //     segments
    // }

    /// A tile whose segments are all within `slice` but may be spaced
    /// The stride should account for the skew
    pub fn new_strided<T: TileConfig>(
        tile_slice: Slice<Line<ES>>,
        #[comptime] stride: u32,
        #[comptime] skew: Skew,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> Tile<ES> {
        comptime! {if let Skew::Element(_) = skew {
            todo!()
        }}

        let (num_segments, segment_length) = Tile::<ES>::segment_info::<T>(ident, config);

        Tile::<ES> {
            slice: tile_slice,
            num_segments,
            segment_length,
            stride,
            skew,
        }
    }

    pub fn as_unlined<T: TileConfig>(
        &self,
        #[comptime] ident: Ident,
        #[comptime] config: T,
    ) -> (Slice<ES>, u32) {
        (
            self.slice.try_cast_unchecked(),
            self.stride * config.stage_line_size(ident),
        )
    }

    pub fn get_segment(&self, segment_index: u32) -> Segment<ES> {
        Segment::<ES> {
            tile_slice: self.slice,
            offset: segment_index * self.stride,
            num_data: self.segment_length,
            skew: comptime!(self.skew),
        }
    }
}
