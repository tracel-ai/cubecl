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
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSlice<S, T> {
    pub fn new(slice: Slice<Line<S>>, #[comptime] line_size: u32) -> ReinterpretSlice<S, T> {
        let source_size = size_of::<S>();
        let target_size = size_of::<T>();
        let optimized = comptime!(optimize_line_size(source_size, target_size, line_size));

        let slice = if optimized != line_size {
            slice.with_line_size(optimized)
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
        let source_size = size_of::<S>();
        let target_size = size_of::<T>();
        let optimized = comptime!(optimize_line_size(source_size, target_size, line_size));
        let slice = if optimized != line_size {
            slice.with_line_size(optimized)
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

fn optimize_line_size(source_size: u32, target_size: u32, line_size: u32) -> u32 {
    let line_source_size = source_size * line_size;

    match line_source_size.cmp(&target_size) {
        std::cmp::Ordering::Less => {
            if target_size % line_source_size != 0 {
                panic!("incompatible number of bytes");
            }
            let ratio = target_size / line_source_size;
            line_size * ratio
        }
        std::cmp::Ordering::Greater => {
            if line_source_size % target_size != 0 {
                panic!("incompatible number of bytes");
            }
            let ratio = line_source_size / target_size;
            line_size / ratio
        }
        std::cmp::Ordering::Equal => line_size,
    }
}
