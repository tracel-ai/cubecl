use core::{cmp::Ordering, marker::PhantomData};

use cubecl::prelude::*;
use cubecl_core as cubecl;

/// This struct allows to take a list of `Line<S>` and reinterpret it
/// as a list of `T`. Semantically, this is equivalent to bitcasting the list of `Line<S>`
/// to a list of `T`. When indexing, the index is valid in the casted list.
#[derive(CubeType)]
pub struct ReinterpretList<S: CubePrimitive, T: CubePrimitive, L: List<Line<S>>> {
    list: L,

    #[cube(comptime)]
    line_size: u32,

    #[cube(comptime)]
    num_bytes_line_source: u32,

    #[cube(comptime)]
    num_bytes_target: u32,

    #[cube(comptime)]
    _phantom: PhantomData<(S, T)>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive, L: List<Line<S>>> ReinterpretList<S, T, L> {
    pub fn new(list: L, #[comptime] line_size: u32) -> ReinterpretList<S, T, L> {
        let num_bytes_line_source = comptime!(core::mem::size_of::<S>() as u32 * line_size);
        let num_bytes_target = comptime!(core::mem::size_of::<T>() as u32);

        comptime! {
            match num_bytes_line_source.cmp(&num_bytes_target) {
                Ordering::Equal => {}
                Ordering::Less => {
                    // TODO: Support fractional encoding (e.g. 3 lines encode 2 values)
                    if num_bytes_target % num_bytes_line_source != 0 {
                        panic!("incompatible number of bytes");
                    }
                }
                Ordering::Greater => {
                    // TODO: Support fractional encoding (e.g. 3 lines encode 4 values)
                    if num_bytes_line_source % num_bytes_target != 0 {
                        panic!("incompatible number of bytes");
                    }
                }
            }
        }

        ReinterpretList::<S, T, L> {
            list,
            line_size,
            num_bytes_line_source,
            num_bytes_target,
            _phantom: PhantomData,
        }
    }

    #[allow(clippy::comparison_chain)]
    pub fn read(&self, index: u32) -> T {
        if comptime!(self.num_bytes_line_source == self.num_bytes_target) {
            T::reinterpret(self.list.read(index))
        } else if comptime!(self.num_bytes_line_source < self.num_bytes_target) {
            self.read_smaller_lines(index)
        } else {
            self.read_larger_lines(index)
        }
    }

    fn read_smaller_lines(&self, index: u32) -> T {
        let num_lines_to_read = comptime!(self.num_bytes_target / self.num_bytes_line_source);

        // This will contains the content of `num_lines_to_read` lines merged
        // into a single larger line.
        let target_line_size = comptime!(num_lines_to_read * self.line_size);
        let mut merged_lines = Line::<S>::empty(target_line_size);

        let first = index * num_lines_to_read;
        #[unroll]
        for i in 0..num_lines_to_read {
            let line = self.list.read(first + i);
            #[unroll]
            for j in 0..self.line_size {
                merged_lines[i * self.line_size + j] = line[j]
            }
        }
        T::reinterpret(merged_lines)
    }

    fn read_larger_lines(&self, index: u32) -> T {
        let num_targets_per_line = comptime!(self.num_bytes_line_source / self.num_bytes_target);

        if comptime!(num_targets_per_line > self.line_size) {
            panic!(
                "the number of bytes in Target ({}) must be a non-zero multiple of the number of bytes in Source ({}).",
                self.num_bytes_target,
                self.num_bytes_line_source / self.line_size,
            );
        }

        let num_elems_to_read = comptime!(self.line_size / num_targets_per_line);

        let line = self.list.read(index / num_targets_per_line);
        let index_in_line = index % num_targets_per_line;

        // A sub-segment of the total line.
        let mut line_segment = Line::empty(num_elems_to_read);

        #[unroll]
        for j in 0..num_elems_to_read {
            line_segment[j] = line[index_in_line * num_elems_to_read + j]
        }
        T::reinterpret(line_segment)
    }
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive, L: ListMut<Line<S>>> ReinterpretList<S, T, L> {
    #[allow(clippy::comparison_chain)]
    pub fn write(&self, index: u32, value: T) {
        let reinterpreted = Line::<S>::reinterpret(value);
        if comptime!(self.num_bytes_line_source == self.num_bytes_target) {
            self.list.write(index, reinterpreted);
        } else if comptime!(self.num_bytes_line_source < self.num_bytes_target) {
            self.write_smaller_lines(index, reinterpreted);
        } else {
            self.write_larger_lines(index, reinterpreted);
        }
    }

    fn write_smaller_lines(&self, index: u32, reinterpreted: Line<S>) {
        let num_lines_to_write = comptime!(self.num_bytes_target / self.num_bytes_line_source);

        let first = index * num_lines_to_write;
        #[unroll]
        for i in 0..num_lines_to_write {
            let mut line = Line::empty(self.line_size);
            #[unroll]
            for j in 0..self.line_size {
                line[j] = reinterpreted[i * self.line_size + j];
            }
            self.list.write(first + i, line);
        }
    }

    fn write_larger_lines(&self, index: u32, reinterpreted: Line<S>) {
        let num_targets_per_line = comptime!(self.num_bytes_line_source / self.num_bytes_target);

        if comptime!(num_targets_per_line > self.line_size) {
            panic!(
                "the number of bytes in Target ({}) must be a non-zero multiple of the number of bytes in Source ({}).",
                self.num_bytes_target,
                self.num_bytes_line_source / self.line_size,
            );
        }

        let num_elems_to_write = comptime!(self.line_size / num_targets_per_line);

        let mut line = self.list.read(index / num_targets_per_line);
        let index_in_line = index % num_targets_per_line;

        #[unroll]
        for j in 0..num_elems_to_write {
            line[index_in_line * num_elems_to_write + j] = reinterpreted[j];
        }

        self.list.write(index / num_targets_per_line, line);
    }
}
