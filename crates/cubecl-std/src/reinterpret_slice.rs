use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, unexpanded};

/// This struct allows to take a slice of `Line<S>` and reinterpret it
/// as a slice of `T`. Semantically, this is equivalent to reinterpreting the slice of `Line<S>`
/// to a slice of `T`. When indexing, the index is valid in the casted list.
///
/// # Warning
///
/// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
#[derive(CubeType)]
pub struct ReinterpretSlice<S: CubePrimitive, T: CubePrimitive> {
    slice: Slice<Line<S>>,

    #[cube(comptime)]
    line_size: u32,

    #[cube(comptime)]
    load_many: Option<u32>,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSlice<S, T> {
    pub fn new(slice: Slice<Line<S>>, #[comptime] line_size: u32) -> ReinterpretSlice<S, T> {
        let source_size = size_of::<S>();
        let target_size = size_of::<T>();
        let (optimized_line_size, load_many) =
            comptime!(optimize_line_size(source_size, line_size, target_size));

        match comptime!(optimized_line_size) {
            Some(line_size) => ReinterpretSlice::<S, T> {
                slice: slice.with_line_size(line_size),
                line_size,
                load_many,
                _phantom: PhantomData,
            },
            None => ReinterpretSlice::<S, T> {
                slice,
                line_size,
                load_many,
                _phantom: PhantomData,
            },
        }
    }

    pub fn read(&self, index: u32) -> T {
        match comptime!(self.load_many) {
            Some(amount) => {
                // panic!("HELLO {} {}", amount, self.line_size);
                let first = index * amount;
                let mut line = Line::empty(comptime!(amount * self.line_size));
                #[unroll]
                for k in 0..amount {
                    let elem = self.slice[first + k];
                    #[unroll]
                    for j in 0..self.line_size {
                        line[k + j] = elem[j];
                    }
                }
                T::reinterpret(line)
            }
            None => {
                // panic!("GOODBYE");
                T::reinterpret(self.slice[index])
            }
        }
    }
}

/// This struct allows to take a mutable slice of `Line<S>` and reinterpret it
/// as a mutable slice of `T`. Semantically, this is equivalent to reinterpreting the slice of `Line<S>`
/// to a mutable slice of `T`. When indexing, the index is valid in the casted list.
///
/// # Warning
///
/// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
#[derive(CubeType)]
pub struct ReinterpretSliceMut<S: CubePrimitive, T: CubePrimitive> {
    slice: SliceMut<Line<S>>,

    #[cube(comptime)]
    load_many: Option<u32>,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSliceMut<S, T> {
    pub fn new(slice: SliceMut<Line<S>>, #[comptime] line_size: u32) -> ReinterpretSliceMut<S, T> {
        let source_size = size_of::<S>();
        let target_size = size_of::<T>();
        let (line_size, load_many) =
            comptime!(optimize_line_size(source_size, line_size, target_size));
        ReinterpretSliceMut::<S, T> {
            slice: match comptime!(line_size) {
                Some(line_size) => slice.with_line_size(line_size),
                None => slice,
            },
            load_many,
            _phantom: PhantomData,
        }
    }

    pub fn read(&self, index: u32) -> T {
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let mut line = Line::empty(amount);
                #[unroll]
                for k in 0..amount {
                    line[k] = self.slice[first + k];
                }
                T::reinterpret(line)
            }
            None => T::reinterpret(self.slice[index]),
        }
    }

    pub fn write(&mut self, index: u32, value: T) {
        let reinterpreted = Line::<S>::reinterpret(value);
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let line_size = comptime!(reinterpreted.size() / amount);

                #[unroll]
                for k in 0..amount {
                    let mut line = Line::empty(line_size);
                    #[unroll]
                    for j in 0..line_size {
                        line[j] = reinterpreted[k * line_size + j];
                    }
                    self.slice[first + k] = line;
                }
            }
            None => self.slice[index] = reinterpreted,
        }
    }
}

fn optimize_line_size(
    source_size: u32,
    line_size: u32,
    target_size: u32,
) -> (Option<u32>, Option<u32>) {
    let line_source_size = source_size * line_size;
    if line_source_size < target_size {
        if target_size % line_source_size != 0 {
            panic!("incompatible number of bytes");
        }

        let ratio = target_size / line_source_size;

        (None, Some(ratio))
    } else if line_source_size > target_size {
        if line_source_size % target_size != 0 {
            panic!("incompatible number of bytes");
        }
        let ratio = line_source_size / target_size;

        (Some(line_size / ratio), None)
    } else {
        (None, None)
    }
}

pub fn size_of<S: CubePrimitive>() -> u32 {
    unexpanded!()
}

pub mod size_of {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<S: CubePrimitive>(context: &mut cubecl::prelude::Scope) -> u32 {
        S::as_elem(context).size() as u32
    }
}
