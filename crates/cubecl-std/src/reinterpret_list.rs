use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

/// This struct allows to take a list of `Line<S>` and reinterpret it
/// as a list of `T`. Semantically, this is equivalent to bitcasting the list of `Line<S>`
/// to a list of `T`. When indexing, the index is valid in the casted list.
///
/// # Warning
///
/// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
#[derive(CubeType)]
pub struct ReinterpretSlice<S: CubePrimitive, T: CubePrimitive> {
    slice: SliceMut<Line<S>>,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSlice<S, T> {
    pub fn new(slice: SliceMut<Line<S>>, #[comptime] line_size: u32) -> ReinterpretSlice<S, T> {
        let num_bytes_line_source = comptime!(core::mem::size_of::<S>() as u32 * line_size);
        let num_bytes_target = comptime!(core::mem::size_of::<T>() as u32);

        let mut line_size = line_size; // I wasn't able to use `mut` directly on the argument.

        let slice = if comptime!(num_bytes_line_source < num_bytes_target) {
            // TODO: Support fractional encoding (e.g. 3 sources encode 2 targets)
            comptime! {
                if num_bytes_target % num_bytes_line_source != 0 {
                    panic!("incompatible number of bytes");
                }
                let ratio = num_bytes_target / num_bytes_line_source;
                line_size *= ratio;
            }
            slice.with_line_size(line_size)
        } else if comptime!(num_bytes_line_source > num_bytes_target) {
            // TODO: Support fractional encoding (e.g. 3 sources encode 4 targets)
            comptime! {
                if num_bytes_line_source % num_bytes_target != 0 {
                    panic!("incompatible number of bytes");
                }
                let ratio = num_bytes_line_source / num_bytes_target;
                line_size /= ratio;
            }
            slice.with_line_size(line_size)
        } else {
            slice
        };

        ReinterpretSlice::<S, T> {
            slice,
            _phantom: PhantomData,
        }
    }

    #[allow(clippy::comparison_chain)]
    pub fn read(&self, index: u32) -> T {
        T::reinterpret(self.slice[index])
    }

    #[allow(clippy::comparison_chain)]
    pub fn write(&mut self, index: u32, value: T) {
        let reinterpreted = Line::<S>::reinterpret(value);
        self.slice[index] = reinterpreted;
    }
}
