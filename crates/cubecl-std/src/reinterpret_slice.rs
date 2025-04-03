use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core as cubecl;

// TODO Move to core
#[derive(Clone)]
pub struct Read;

#[derive(Clone)]
pub struct ReadWrite;

pub trait CubeIO {}

impl CubeIO for Read {}
impl CubeIO for ReadWrite {}

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
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSlice<S, T> {
    pub fn new(slice: Slice<Line<S>>, #[comptime] line_size: u32) -> ReinterpretSlice<S, T> {
        let optimized = comptime!(optimize_line_size::<S, T>(line_size));
        let slice = if optimized != line_size {
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

    // #[allow(clippy::comparison_chain)]
    // pub fn write(&mut self, index: u32, value: T) {
    //     let reinterpreted = Line::<S>::reinterpret(value);
    //     self.slice[index] = reinterpreted;
    // }
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
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSliceMut<S, T> {
    pub fn new(slice: SliceMut<Line<S>>, #[comptime] line_size: u32) -> ReinterpretSliceMut<S, T> {
        let optimized = comptime!(optimize_line_size::<S, T>(line_size));
        let slice = if optimized != line_size {
            slice.with_line_size(line_size)
        } else {
            slice
        };
        ReinterpretSliceMut::<S, T> {
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

fn optimize_line_size<S: CubePrimitive, T: CubePrimitive>(line_size: u32) -> u32 {
    let num_bytes_line_source = core::mem::size_of::<S>() as u32 * line_size;
    let num_bytes_target = core::mem::size_of::<T>() as u32;

    match num_bytes_line_source.cmp(&num_bytes_target) {
        std::cmp::Ordering::Less => {
            if num_bytes_target % num_bytes_line_source != 0 {
                panic!("incompatible number of bytes");
            }
            let ratio = num_bytes_target / num_bytes_line_source;
            line_size * ratio
        }
        std::cmp::Ordering::Equal => {
            if num_bytes_line_source % num_bytes_target != 0 {
                panic!("incompatible number of bytes");
            }
            let ratio = num_bytes_line_source / num_bytes_target;
            line_size / ratio
        }
        std::cmp::Ordering::Greater => line_size,
    }
}
