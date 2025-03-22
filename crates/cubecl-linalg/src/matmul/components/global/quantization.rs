use cubecl::prelude::*;
use cubecl_core as cubecl;

/// Store the quantization meta-parameters.
/// For now, we only support symmetric quantization,
/// thus we only store the scaling.
#[derive(CubeType, Clone, Copy)]
pub struct Quantization<EG: Numeric> {
    pub lhs: Slice<Line<EG>>,
    pub rhs: Slice<Line<EG>>,
    pub out: SliceMut<Line<EG>>,
}

#[cube]
impl<EG: Numeric> Quantization<EG> {
    pub fn read_scale_lhs(&self, index: u32, #[comptime] line_size: u32) -> f32 {
        read_f32(self.lhs, index, line_size)
    }

    pub fn read_scale_rhs(&self, index: u32, #[comptime] line_size: u32) -> f32 {
        read_f32(self.rhs, index, line_size)
    }

    pub fn write_scale_out(self, index: u32, value: f32, #[comptime] line_size: u32) {
        write_f32(self.out, index, value, line_size)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct IndexRange {
    pub current: u32,
    pub end: u32,
    pub step: u32, // TODO needed?
}

#[cube]
impl IndexRange {
    pub fn advance(&mut self) {
        self.current += self.step;
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct IndexedQuantization<EG: Numeric> {
    pub quantization: Quantization<EG>,
    pub range_lhs: IndexRange,
    pub range_rhs: IndexRange,
    pub index_out: u32,
}

#[cube]
impl<EG: Numeric> IndexedQuantization<EG> {
    pub fn new(
        quantization: Quantization<EG>,
        range_lhs: IndexRange,
        range_rhs: IndexRange,
        index_out: u32,
    ) -> IndexedQuantization<EG> {
        IndexedQuantization::<EG> {
            quantization,
            range_lhs,
            range_rhs,
            index_out,
        }
    }

    pub fn read_current_scale_lhs(&self, #[comptime] line_size: u32) -> f32 {
        let index = self.range_lhs.current;
        self.quantization.read_scale_lhs(index, line_size)
    }

    pub fn read_current_scale_rhs(&self, #[comptime] line_size: u32) -> f32 {
        let index = self.range_rhs.current;
        self.quantization.read_scale_rhs(index, line_size)
    }

    pub fn write_scale_out(&mut self, value: f32, #[comptime] line_size: u32) {
        let index = self.index_out;
        self.quantization.write_scale_out(index, value, line_size);
    }

    pub fn advance_indices(&mut self) {
        self.range_lhs.advance();
        self.range_rhs.advance();
    }
}

/// This functions assume that `slice` actually store f32 values,
/// but because of typing issue, it is represented in the type system as Slice<Line<EG>>.
/// This reads and converts to an f32 the value at position `index` in the slice, that is bytes `4 * index`  to `4 * (index + 1)`
/// when viewing `slice` as an array of bytes.
#[cube]
fn read_f32<EG: Numeric>(slice: Slice<Line<EG>>, index: u32, #[comptime] line_size: u32) -> f32 {
    let num_bytes_line_eg = comptime!(core::mem::size_of::<EG>() as u32) * line_size;
    match num_bytes_line_eg {
        1 => {
            // Each item is a 1 byte value, we need four of them.
            let start = index * 4;
            let mut bytes = Line::empty(4);
            #[unroll]
            for k in 0..4 {
                bytes[k] = slice[start + k][0];
            }
            f32::bitcast_from(bytes)
        }
        2 => {
            // Each item is a 2 bytes value, we need two of them.
            let start = index * 2;
            let mut bytes = Line::empty(2); // This is either a line of two lines of u8 / i8 or a line of two f16 / u16 / i16.
            #[unroll]
            for k in 0..2 {
                bytes[k] = slice[start + k][0];
            }
            f32::bitcast_from(bytes)
        }
        4 => f32::bitcast_from(slice[index]), // Each item is a 4 bytes value, we need one of them.
        8 => {
            // Each item is a 8 bytes value, we need half of one.
            let outer = index / 2;
            let inner = index % 2;
            let mut bytes = Line::empty(line_size / 2);
            #[unroll]
            for k in 0..line_size / 2 {
                bytes[k] = slice[outer][inner + k];
            }
            f32::bitcast_from(bytes)
        }
        16 => {
            // Each item is a 16 bytes value, we need a quarter of one.
            let outer = index / 4;
            let inner = index % 4;
            let mut bytes = Line::empty(line_size / 4);
            #[unroll]
            for k in 0..line_size / 4 {
                bytes[k] = slice[outer][inner + k];
            }
            f32::bitcast_from(bytes)
        }
        32 => {
            // Each item is a 32 bytes value, we need an eight of one.
            let outer = index / 8;
            let inner = index % 8;
            let mut bytes = Line::empty(line_size / 8);
            #[unroll]
            for k in 0..line_size / 8 {
                bytes[k] = slice[outer][inner + k];
            }
            f32::bitcast_from(bytes)
        }
        _ => comptime!(panic!("invalid number of bytes")), // unreachable
    }
}

/// This functions assume that `slice` actually store f32 values,
/// but because of typing issue, it is represented in the type system as Slice<Line<EG>>.
/// This writes the given `value` at position `index` in the slice, that is bytes `4 * index`  to `4 * (index + 1)`
/// when viewing `slice` as an array of bytes.
#[cube]
fn write_f32<EG: Numeric>(
    mut slice: SliceMut<Line<EG>>,
    index: u32,
    value: f32,
    #[comptime] line_size: u32,
) {
    let bytes = Line::<EG>::bitcast_from(value);
    let num_bytes_line_eg = comptime!(core::mem::size_of::<EG>() as u32) * line_size;
    match num_bytes_line_eg {
        1 => {
            // Each item is a 1 byte value, we need for of them.
            let start = index * 4;
            #[unroll]
            for k in 0..4 {
                slice[start + k] = Line::new(bytes[k]);
            }
        }
        2 => {
            // Each item is a 2 bytes value, we need two of them.
            let start = index * 2;
            #[unroll]
            for k in 0..2 {
                // We build a line from the first / last two bytes.
                let mut line = Line::<EG>::empty(2);
                line[0] = bytes[2 * k];
                line[1] = bytes[2 * k + 1];

                slice[start + k] = Line::<EG>::bitcast_from(line);
            }
        }
        4 => slice[index] = Line::<EG>::bitcast_from(value), // Each item is a 4 bytes value, we need one of them.
        8 => {
            // Each item is a 8 bytes value, we need half of one.
            let outer = index / 2;
            let inner = index % 2;

            // We convert the item to a pair of f32.
            // Then we overwrite one of the two f32 by the given value.
            // Finally, we convert back to a Line<EG> that we write in the slice.
            let mut line = Line::<f32>::bitcast_from(slice[outer]);
            line[inner] = value;
            slice[outer] = Line::<EG>::bitcast_from(line);
        }
        16 => {
            // Each item is a 16 bytes value, we need a quarter of one.
            let outer = index / 4;
            let inner = index % 4;

            // We convert the item to four f32.
            // Then we overwrite one of the four f32 by the given value.
            // Finally, we convert back to a Line<EG> that we write in the slice.
            let mut line = Line::<f32>::bitcast_from(slice[outer]);
            line[inner] = value;
            slice[outer] = Line::<EG>::bitcast_from(line);
        }
        32 => {
            // Each item is a 32 bytes value, we need an eight of one.
            let outer = index / 8;
            let inner = index % 8;

            // We convert the item to eight f32.
            // Then we overwrite one of the eight f32 by the given value.
            // Finally, we convert back to a Line<EG> that we write in the slice.
            let mut line = Line::<f32>::bitcast_from(slice[outer]);
            line[inner] = value;
            slice[outer] = Line::<EG>::bitcast_from(line);
        }
        _ => comptime!(panic!("invalid number of bytes")), // unreachable
    }
}
