//! A shared-memory destination planes add into atomically, and a tile that reads the sum back.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, ReadsLines, WriteOnly,
    WritesLines,
};

use crate::*;

/// A cube's shared-memory accumulator over one box, summing in the element it was opened with,
/// read back at the element of the tile it was opened over.
#[derive(CubeType)]
pub struct SmemAccumulation {
    /// Writes add into the cell atomically.
    pub sink: Tile,
    /// Reads the sum back at the opened tile's element; valid only once every plane has drained
    /// and the cube has synced.
    pub source: Tile,
}

#[cube]
impl Tile {
    /// A zeroed shared-memory accumulator over this tile's box, summing in `A`, with the cube
    /// synchronized on both sides of the zeroing.
    pub fn smem_accumulation<A: Numeric>(&self) -> SmemAccumulation {
        let define!(Served) = self.element;
        let space = comptime!(self.place.space.clone());
        let elem_bytes = A::size().comptime();
        let form = comptime!(StageForm::dense(
            &space,
            1,
            StageStorage::Strided,
            LineBytes(elem_bytes)
        ));
        let cells = comptime!(form.cells());
        let values = Shared::<[Atomic<A>]>::new_slice(cells);
        // The previous open from this call site may still be read; let it finish first.
        sync_cube();
        let mut cell = UNIT_POS as usize;
        while cell < cells {
            values[cell].store(A::from_int(0));
            cell += CUBE_DIM as usize;
        }
        sync_cube();

        let units = self.units();
        let sink = Memory::smem_backed::<A>(
            comptime!(space.clone()),
            1usize,
            comptime!(FillUnits::cube(units)),
            Backing::new_WriteCall(RawCall::of::<A>(
                ErasedTensor::<A, WriteOnly>::of_smem_accumulate(&values),
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Accumulate),
            comptime!(Overhang::Masked),
        );
        let source = Memory::smem_backed::<Served>(
            space,
            1usize,
            comptime!(FillUnits::cube(units)),
            Backing::new_ReadCall(RawCall::of::<Served>(
                ErasedTensor::<Served, ReadOnly>::of_smem_load::<A>(&values),
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Replace),
            comptime!(Overhang::Never),
        );
        SmemAccumulation {
            sink: sink.nested_like(self),
            source: source.nested_like(self),
        }
    }
}

/// The erased tensors over a shared buffer of atomics.
pub(crate) trait SmemAccumulateSink<E: Numeric> {
    /// The sink that adds into `values`, one scalar per line.
    fn of_smem_accumulate(_values: &Shared<[Atomic<E>]>) -> ErasedTensor<E, WriteOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_accumulate(
        _scope: &Scope,
        values: &<Shared<[Atomic<E>]> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(SmemAccumulate::<E> {
            values: ExpandTypeClone::clone_unchecked(values),
            _e: PhantomData,
        })
    }
}

impl<E: Numeric> SmemAccumulateSink<E> for ErasedTensor<E, WriteOnly> {}

/// The source that loads out of a buffer of atomics summing in `A`, served as `T`.
pub(crate) trait SmemLoadSource<T: Numeric> {
    /// The source that loads `values`, cast to `T`, one scalar per line.
    fn of_smem_load<A: Numeric>(_values: &Shared<[Atomic<A>]>) -> ErasedTensor<T, ReadOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_load<A: Numeric>(
        _scope: &Scope,
        values: &<Shared<[Atomic<A>]> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<T, ReadOnly> {
        ErasedTensorExpand::new(SmemLoad::<A, T> {
            values: ExpandTypeClone::clone_unchecked(values),
            _t: PhantomData,
        })
    }
}

impl<T: Numeric> SmemLoadSource<T> for ErasedTensor<T, ReadOnly> {}

/// A backing that adds into a shared buffer of `Atomic<E>`. Writes and never reads.
struct SmemAccumulate<E: Numeric> {
    values: <Shared<[Atomic<E>]> as CubeType>::ExpandType,
    _e: PhantomData<E>,
}

impl<E: Numeric> ErasedTensorOperationsExpand<E> for SmemAccumulate<E> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        1
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.values.__expand_len_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        accumulate_cell::expand::<E>(scope, &self.values, index, value.into());
    }
}

impl<E: Numeric> WritesLines<E> for SmemAccumulate<E> {}

/// A backing that loads out of a shared buffer of `Atomic<A>`, casting to `T`.
struct SmemLoad<A: Numeric, T: Numeric> {
    values: <Shared<[Atomic<A>]> as CubeType>::ExpandType,
    _t: PhantomData<T>,
}

impl<A: Numeric, T: Numeric> ErasedTensorOperationsExpand<T> for SmemLoad<A, T> {
    fn __expand_vector_size_method(&self, _scope: &Scope) -> VectorSize {
        1
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        self.values.__expand_len_method(scope)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        load_cell::expand::<A, T>(scope, &self.values, index).expand
    }
}

impl<A: Numeric, T: Numeric> ReadsLines<T> for SmemLoad<A, T> {}

/// Add one line, one scalar wide, into its cell.
#[cube]
fn accumulate_cell<E: Numeric>(
    values: &Shared<[Atomic<E>]>,
    index: usize,
    value: Vector<E, Const<1>>,
) {
    values[index].fetch_add(value.extract(0usize));
}

/// Load one cell as a line one scalar wide, cast to `T`.
#[cube]
fn load_cell<A: Numeric, T: Numeric>(
    values: &Shared<[Atomic<A>]>,
    index: usize,
) -> Vector<T, Const<1>> {
    Vector::new(T::cast_from(values[index].load()))
}
