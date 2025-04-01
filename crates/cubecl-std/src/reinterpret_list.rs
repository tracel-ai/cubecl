use core::{cmp::Ordering, marker::PhantomData};

use cubecl::prelude::*;
use cubecl_core as cubecl;

#[derive(CubeType)]
pub struct ReinterpretList<From: CubePrimitive, To: CubePrimitive, L: List<Line<From>>> {
    list: L,

    #[cube(comptime)]
    line_size: u32,

    #[cube(comptime)]
    num_bytes_line_from: u32,

    #[cube(comptime)]
    num_bytes_to: u32,

    #[cube(comptime)]
    _phantom: PhantomData<(From, To)>,
}

#[cube]
impl<From: CubePrimitive, To: CubePrimitive, L: List<Line<From>>> ReinterpretList<From, To, L> {
    pub fn new(list: L, #[comptime] line_size: u32) -> ReinterpretList<From, To, L> {
        let num_bytes_line_from = comptime!(core::mem::size_of::<From>() as u32 * line_size);
        let num_bytes_to = comptime!(core::mem::size_of::<To>() as u32);
        // panic!("FROM {num_bytes_line_from} TO {num_bytes_to}");

        comptime! {
            match num_bytes_line_from.cmp(&num_bytes_to) {
                Ordering::Equal => {}
                Ordering::Less => {
                    // TODO: Support fractional encoding (e.g. 3 lines encode 2 values)
                    if num_bytes_to % num_bytes_line_from != 0 {
                        panic!("incompatible number of bytes");
                    }
                }
                Ordering::Greater => {
                    // TODO: Support fractional encoding (e.g. 3 lines encode 4 values)
                    if num_bytes_line_from % num_bytes_to != 0 {
                        panic!("incompatible number of bytes");
                    }
                }
            }
        }

        ReinterpretList::<From, To, L> {
            list,
            line_size,
            num_bytes_line_from,
            num_bytes_to,
            _phantom: PhantomData,
        }
    }

    #[allow(clippy::comparison_chain)]
    pub fn read(&self, index: u32) -> To {
        if comptime!(self.num_bytes_line_from == self.num_bytes_to) {
            // panic!("PATH 1");
            To::bitcast_from(self.list.read(index))
        } else if comptime!(self.num_bytes_line_from < self.num_bytes_to) {
            self.read_smaller_lines(index)
        } else {
            self.read_larger_lines(index)
        }
    }

    pub fn read_smaller_lines(&self, index: u32) -> To {
        let num_lines_to_read = comptime!(self.num_bytes_to / self.num_bytes_line_from);

        // This will contains the content of `num_lines_to_read` lines merged
        // into a single larger line.
        let target_line_size = comptime!(num_lines_to_read * self.line_size);
        let mut merged_lines = Line::<From>::empty(target_line_size);

        let first = index * num_lines_to_read;
        #[unroll]
        for i in 0..num_lines_to_read {
            let line = self.list.read(first + i);
            #[unroll]
            for j in 0..self.line_size {
                merged_lines[i * self.line_size + j] = line[j]
            }
        }
        To::bitcast_from(merged_lines)
    }

    pub fn read_larger_lines(&self, index: u32) -> To {
        let num_outputs_per_line = comptime!(self.num_bytes_line_from / self.num_bytes_to);

        if comptime!(num_outputs_per_line > self.line_size) {
            panic!(
                "the number of bytes in To ({}) must be a non-zero multiple of the number of bytes in From ({})-> {}, {}",
                self.num_bytes_to,
                self.num_bytes_line_from / self.line_size,
                self.num_bytes_line_from,
                num_outputs_per_line
            );
        }

        let num_elems_to_read = comptime!(self.line_size / num_outputs_per_line);

        let line = self.list.read(index / num_outputs_per_line);
        let index_in_line = index % num_outputs_per_line;

        // A sub-segment of the total line.
        let mut line_segment = Line::empty(num_elems_to_read);

        #[unroll]
        for j in 0..num_elems_to_read {
            line_segment[j] = line[index_in_line * num_elems_to_read + j]
        }
        To::bitcast_from(line_segment)
    }
}
