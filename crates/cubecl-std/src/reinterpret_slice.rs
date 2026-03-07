use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, ir::LineSize, unexpanded};

/// This struct allows to take a slice of `Line<S>` and reinterpret it
/// as a slice of `T`. Semantically, this is equivalent to reinterpreting the slice of `Line<S>`
/// to a slice of `T`. When indexing, the index is valid in the casted list.
///
/// # Warning
///
/// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
#[derive(CubeType)]
pub struct ReinterpretSlice<S: CubePrimitive, T: CubePrimitive> {
    // Dummy line size for downcasting later
    slice: Slice<Line<S, Const<0>>>,

    #[cube(comptime)]
    line_size: LineSize,

    #[cube(comptime)]
    load_many: Option<usize>,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSlice<S, T> {
    pub fn new<N: Size>(slice: Slice<Line<S, N>>) -> ReinterpretSlice<S, T> {
        let line_size = N::value();
        let source_size = size_of::<S>();
        let target_size = size_of::<T>();
        let (optimized_line_size, load_many) =
            comptime!(optimize_line_size(source_size, line_size, target_size));
        match comptime!(optimized_line_size) {
            Some(line_size) => {
                let size!(N2) = line_size;
                ReinterpretSlice::<S, T> {
                    slice: slice.with_line_size::<N2>().downcast(),
                    line_size,
                    load_many,
                    _phantom: PhantomData,
                }
            }
            None => ReinterpretSlice::<S, T> {
                slice: slice.downcast(),
                line_size,
                load_many,
                _phantom: PhantomData,
            },
        }
    }

    pub fn read(&self, index: usize) -> T {
        let size!(N) = self.line_size;
        let slice: Slice<Line<S, N>> = self.slice.downcast();
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let size!(N2) = comptime!(amount * self.line_size);
                let mut line = Line::<S, N2>::empty();
                #[unroll]
                for k in 0..amount {
                    let elem = slice[first + k];
                    #[unroll]
                    for j in 0..self.line_size {
                        line[k * self.line_size + j] = elem[j];
                    }
                }
                T::reinterpret(line)
            }
            None => T::reinterpret(slice[index]),
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
    slice: SliceMut<Line<S, Const<0>>>,

    #[cube(comptime)]
    line_size: LineSize,

    #[cube(comptime)]
    load_many: Option<usize>,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSliceMut<S, T> {
    pub fn new<N: Size>(slice: SliceMut<Line<S, N>>) -> ReinterpretSliceMut<S, T> {
        let line_size = N::value();
        let source_size = size_of::<S>();
        let target_size = size_of::<T>();
        let (optimized_line_size, load_many) =
            comptime!(optimize_line_size(source_size, line_size, target_size));
        match comptime!(optimized_line_size) {
            Some(line_size) => {
                let size!(N2) = line_size;
                ReinterpretSliceMut::<S, T> {
                    slice: slice.with_line_size::<N2>().downcast(),
                    line_size,
                    load_many,
                    _phantom: PhantomData,
                }
            }
            None => ReinterpretSliceMut::<S, T> {
                slice: slice.downcast(),
                line_size,
                load_many,
                _phantom: PhantomData,
            },
        }
    }

    pub fn read(&self, index: usize) -> T {
        let size!(N) = self.line_size;
        let slice: Slice<Line<S, N>, ReadWrite> = self.slice.downcast();
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let size!(N2) = comptime!(amount * self.line_size);
                let mut line = Line::<S, N2>::empty();
                #[unroll]
                for k in 0..amount {
                    let elem = slice[first + k];
                    #[unroll]
                    for j in 0..self.line_size {
                        line[k * self.line_size + j] = elem[j];
                    }
                }
                T::reinterpret(line)
            }
            None => T::reinterpret(slice[index]),
        }
    }

    pub fn write(&mut self, index: usize, value: T) {
        let size!(N) = self.line_size;
        let mut slice: Slice<Line<S, N>, ReadWrite> = self.slice.downcast();
        let size!(N1) = reinterpret_line_size::<T, S>(&value);
        let reinterpreted = Line::<S, N1>::reinterpret(value);
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let line_size = comptime!(reinterpreted.size() / amount);

                #[unroll]
                for k in 0..amount {
                    let mut line = Line::empty();
                    #[unroll]
                    for j in 0..line_size {
                        line[j] = reinterpreted[k * line_size + j];
                    }
                    slice[first + k] = line;
                }
            }
            None => slice[index] = Line::cast_from(reinterpreted),
        }
    }
}

fn optimize_line_size(
    source_size: u32,
    line_size: LineSize,
    target_size: u32,
) -> (Option<usize>, Option<usize>) {
    let target_size = target_size as usize;
    let line_source_size = source_size as usize * line_size;
    match line_source_size.cmp(&target_size) {
        core::cmp::Ordering::Less => {
            if !target_size.is_multiple_of(line_source_size) {
                panic!("incompatible number of bytes");
            }

            let ratio = target_size / line_source_size;

            (None, Some(ratio))
        }
        core::cmp::Ordering::Greater => {
            if !line_source_size.is_multiple_of(target_size) {
                panic!("incompatible number of bytes");
            }
            let ratio = line_source_size / target_size;

            (Some(line_size / ratio), None)
        }
        core::cmp::Ordering::Equal => (None, None),
    }
}

pub fn size_of<S: CubePrimitive>() -> u32 {
    unexpanded!()
}

pub mod size_of {
    use super::*;
    #[allow(unused, clippy::all)]
    pub fn expand<S: CubePrimitive>(context: &mut cubecl::prelude::Scope) -> u32 {
        S::as_type(context).size() as u32
    }
}
