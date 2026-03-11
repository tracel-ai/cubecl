use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, ir::VectorSize, unexpanded};

/// This struct allows to take a slice of `Vector<S>` and reinterpret it
/// as a slice of `T`. Semantically, this is equivalent to reinterpreting the slice of `Vector<S>`
/// to a slice of `T`. When indexing, the index is valid in the casted list.
///
/// # Warning
///
/// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
#[derive(CubeType)]
pub struct ReinterpretSlice<S: CubePrimitive, T: CubePrimitive> {
    // Dummy vector size for downcasting later
    slice: Slice<S>,

    #[cube(comptime)]
    vector_size: VectorSize,

    #[cube(comptime)]
    load_many: Option<usize>,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSlice<S, T> {
    pub fn new(slice: Slice<S>) -> ReinterpretSlice<S, T> {
        let in_vector_size = slice.vector_size();
        let source_size = S::Scalar::type_size();
        let target_size = T::Scalar::type_size();
        let (optimized_vector_size, load_many) = comptime!(optimize_vector_size(
            source_size,
            in_vector_size,
            target_size
        ));
        match comptime!(optimized_vector_size) {
            Some(vector_size) => {
                let size!(N2) = vector_size;
                let slice = slice.into_vectorized().with_vector_size::<N2>();

                ReinterpretSlice::<S, T> {
                    slice: unsafe { slice.downcast_unchecked() },
                    vector_size,
                    load_many,
                    _phantom: PhantomData,
                }
            }
            None => ReinterpretSlice::<S, T> {
                slice,
                vector_size: in_vector_size,
                load_many,
                _phantom: PhantomData,
            },
        }
    }

    pub fn read(&self, index: usize) -> T {
        let size!(N) = self.vector_size;
        let slice = self.slice.into_vectorized().with_vector_size::<N>();
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let size!(N2) = comptime!(amount * self.vector_size);
                let mut vector = Vector::<S::Scalar, N2>::empty();
                #[unroll]
                for k in 0..amount {
                    let elem = slice[first + k];
                    #[unroll]
                    for j in 0..self.vector_size {
                        vector[k * self.vector_size + j] = elem[j];
                    }
                }
                T::reinterpret(vector)
            }
            None => T::reinterpret(slice[index]),
        }
    }
}

/// This struct allows to take a mutable slice of `Vector<S>` and reinterpret it
/// as a mutable slice of `T`. Semantically, this is equivalent to reinterpreting the slice of `Vector<S>`
/// to a mutable slice of `T`. When indexing, the index is valid in the casted list.
///
/// # Warning
///
/// Currently, this only work with `cube(launch_unchecked)` and is not supported on wgpu.
#[derive(CubeType)]
pub struct ReinterpretSliceMut<S: CubePrimitive, T: CubePrimitive> {
    slice: SliceMut<S>,

    #[cube(comptime)]
    vector_size: VectorSize,

    #[cube(comptime)]
    load_many: Option<usize>,

    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

#[cube]
impl<S: CubePrimitive, T: CubePrimitive> ReinterpretSliceMut<S, T> {
    pub fn new(slice: SliceMut<S>) -> ReinterpretSliceMut<S, T> {
        let in_vector_size = slice.vector_size();
        let source_size = S::Scalar::type_size();
        let target_size = T::Scalar::type_size();
        let (optimized_vector_size, load_many) = comptime!(optimize_vector_size(
            source_size,
            in_vector_size,
            target_size
        ));
        match comptime!(optimized_vector_size) {
            Some(vector_size) => {
                let size!(N2) = vector_size;
                let slice = slice.into_vectorized().with_vector_size::<N2>();

                ReinterpretSliceMut::<S, T> {
                    slice: unsafe { slice.downcast_unchecked() },
                    vector_size,
                    load_many,
                    _phantom: PhantomData,
                }
            }
            None => ReinterpretSliceMut::<S, T> {
                slice,
                vector_size: in_vector_size,
                load_many,
                _phantom: PhantomData,
            },
        }
    }

    pub fn read(&self, index: usize) -> T {
        let size!(N) = self.vector_size;
        let slice = self.slice.into_vectorized().with_vector_size::<N>();
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let size!(N2) = comptime!(amount * self.vector_size);
                let mut vector = Vector::<S::Scalar, N2>::empty();
                #[unroll]
                for k in 0..amount {
                    let elem = slice[first + k];
                    #[unroll]
                    for j in 0..self.vector_size {
                        vector[k * self.vector_size + j] = elem[j];
                    }
                }
                T::reinterpret(vector)
            }
            None => T::reinterpret(slice[index]),
        }
    }

    pub fn write(&mut self, index: usize, value: T) {
        let size!(N) = self.vector_size;
        let mut slice = self.slice.into_vectorized().with_vector_size::<N>();
        let size!(N1) = S::reinterpret_vectorization::<T>();
        let reinterpreted = Vector::<S::Scalar, N1>::reinterpret(value);
        match comptime!(self.load_many) {
            Some(amount) => {
                let first = index * amount;
                let vector_size = comptime!(reinterpreted.size() / amount);

                #[unroll]
                for k in 0..amount {
                    let mut vector = Vector::empty();
                    #[unroll]
                    for j in 0..vector_size {
                        vector[j] = reinterpreted[k * vector_size + j];
                    }
                    slice[first + k] = vector;
                }
            }
            None => slice[index] = Vector::cast_from(reinterpreted),
        }
    }
}

fn optimize_vector_size(
    source_size: usize,
    vector_size: VectorSize,
    target_size: usize,
) -> (Option<usize>, Option<usize>) {
    let vector_source_size = source_size * vector_size;
    match vector_source_size.cmp(&target_size) {
        core::cmp::Ordering::Less => {
            if !target_size.is_multiple_of(vector_source_size) {
                panic!("incompatible number of bytes");
            }

            let ratio = target_size / vector_source_size;

            (None, Some(ratio))
        }
        core::cmp::Ordering::Greater => {
            if !vector_source_size.is_multiple_of(target_size) {
                panic!("incompatible number of bytes");
            }
            let ratio = vector_source_size / target_size;

            (Some(vector_size / ratio), None)
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
