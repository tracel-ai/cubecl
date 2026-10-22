//! A shared-memory destination the planes of a cube write exclusively, on the cyclic schedule, and
//! a tile that reads the sum back.
//!
//! [`SmemAccumulation`](super::smem_accumulation::SmemAccumulation)'s writers add into every cell
//! at once through atomics, in whatever order they arrive, into a buffer zeroed first. Here each
//! writer's fragments are dealt into as many chunks as there are writers, by their place in the
//! fragment grid, and the drain runs in as many rounds: in round `r`, writer `w` lands its
//! fragments of chunk `(w + r) mod writers` alone, replacing their cells in round zero and adding
//! into them after, with a plain read and write. Every writer works every round, on cells no other
//! writer touches, a cube barrier between rounds, and drains each fragment once: no atomics, no
//! zeroing, and each cell takes its partials in a fixed order, so the sum is the same bits from run
//! to run.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, ReadsLines, WriteOnly,
    WritesLines,
};

use crate::*;

/// A cube's shared-memory accumulator over one box, summing in `A` and read back at the element
/// of the tile it was opened over, that `writers` writers reach exclusively, on the
/// [`Cyclic`](Schedule::Cyclic) schedule.
///
/// A writer drains its partial with [`drain`](SmemCyclicAccumulation::drain) (or
/// [`drain_at`](SmemCyclicAccumulation::drain_at) for a partial over one window), which runs every
/// round and synchronizes the cube after each; once every writer has, [`source`] holds the sum.
/// Every unit of the cube reaches the drain, each plane with the partial and window it holds.
///
/// [`source`]: SmemCyclicAccumulation::source
#[derive(CubeType)]
pub struct SmemCyclicAccumulation<A: Numeric> {
    values: Shared<[A]>,
    /// The box the accumulator was opened over, which every round's sink is shaped like.
    like: Tile,
    /// Reads the sum back at the opened tile's element; valid only once every writer has drained.
    pub source: Tile,
    #[cube(comptime)]
    writers: usize,
}

#[cube]
impl Tile {
    /// A shared-memory accumulator over this tile's box, summing in `A`, that `writers` writers
    /// reach exclusively, one chunk each a round. Nothing is zeroed: every cell's first writer
    /// replaces it. It is served in lines as wide as this tile's, so a partial drains into it and
    /// its sum lands in this tile line for line.
    pub fn smem_cyclic_accumulation<A: Numeric>(
        &self,
        #[comptime] writers: usize,
    ) -> SmemCyclicAccumulation<A> {
        let define!(Served) = self.element;
        let space = comptime!(self.place.space.clone());
        let width = self.vector_size();
        let size!(W) = width;
        let form = comptime!(StageForm::dense(
            &space,
            width,
            StageStorage::Strided,
            LineBytes(A::size().comptime() * width)
        ));
        let values = Shared::<[A]>::new_slice(comptime!(form.cells() * width));
        // The previous open from this call site may still be read; let it finish first.
        sync_cube();

        let units = self.units();
        let source = Memory::smem_backed::<Served>(
            space,
            width,
            comptime!(FillUnits::cube(units)),
            Backing::new_ReadCall(RawCall::of::<Served>(
                ErasedTensor::<Served, ReadOnly>::of_smem_plain_load::<A, W>(&values),
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Replace),
            comptime!(Overhang::Never),
        );
        SmemCyclicAccumulation::<A> {
            values,
            like: self.clone(),
            source: source.nested_like(self),
            writers,
        }
    }
}

#[cube]
impl<A: Numeric> SmemCyclicAccumulation<A> {
    /// The sink every writer drains into in `round`: it replaces each cell it is handed in round
    /// zero and adds into it after.
    fn sink(&self, round: usize) -> Tile {
        let space = comptime!(self.like.place.space.clone());
        let width = self.like.vector_size();
        let size!(W) = width;
        let form = comptime!(StageForm::dense(
            &space,
            width,
            StageStorage::Strided,
            LineBytes(A::size().comptime() * width)
        ));
        let units = self.like.units();
        let sink = Memory::smem_backed::<A>(
            space,
            width,
            comptime!(FillUnits::cube(units)),
            Backing::new_WriteCall(RawCall::of::<A>(
                ErasedTensor::<A, WriteOnly>::of_smem_cyclic::<W>(&self.values, round == 0),
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Exclusive(Schedule::Cyclic)),
            comptime!(Overhang::Masked),
        );
        sink.nested_like(&self.like)
    }

    /// Drain `partial`, an accumulator over the whole box, every round of the schedule, the cube
    /// synchronized after each: each unit lands its plane's fragments of the round's chunk as
    /// `writer`, its plane's place among the writers. Once every writer of the cube has, the sum is
    /// in [`source`](SmemCyclicAccumulation::source).
    pub fn drain(&self, partial: &Tile, writer: usize) {
        let define!(Served) = partial.element;
        #[unroll]
        for round in 0..self.writers {
            let turn = (writer + round) % comptime!(self.writers);
            partial.drained_chunk_into::<Served, A>(
                &self.sink(round),
                turn,
                comptime!(self.writers),
            );
            sync_cube();
        }
    }

    /// [`drain`](SmemCyclicAccumulation::drain) for a partial over one `window` of the box, as a
    /// kernel that walks its planes itself holds one.
    pub fn drain_at(&self, partial: &Tile, window: &Region, writer: usize) {
        let define!(Served) = partial.element;
        #[unroll]
        for round in 0..self.writers {
            let turn = (writer + round) % comptime!(self.writers);
            partial.drained_chunk_into::<Served, A>(
                &self.sink(round).at(window),
                turn,
                comptime!(self.writers),
            );
            sync_cube();
        }
    }
}

/// The erased tensors over a shared buffer of plain values.
pub(crate) trait SmemCyclicSink<E: Numeric> {
    /// The sink that writes into `values` in lines `W` scalars wide, replacing each cell where
    /// `first` and adding into it otherwise.
    fn of_smem_cyclic<W: Size>(_values: &Shared<[E]>, _first: bool) -> ErasedTensor<E, WriteOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_cyclic<W: Size>(
        _scope: &Scope,
        values: &<Shared<[E]> as CubeType>::ExpandType,
        first: NativeExpand<bool>,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(SmemCyclic::<E, W> {
            values: ExpandTypeClone::clone_unchecked(values),
            first,
            _e: PhantomData,
        })
    }
}

impl<E: Numeric> SmemCyclicSink<E> for ErasedTensor<E, WriteOnly> {}

/// The source that loads out of a shared buffer of plain `A`, served as `T`.
pub(crate) trait SmemPlainLoadSource<T: Numeric> {
    /// The source that loads `values`, cast to `T`, in lines `W` scalars wide.
    fn of_smem_plain_load<A: Numeric, W: Size>(_values: &Shared<[A]>) -> ErasedTensor<T, ReadOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_plain_load<A: Numeric, W: Size>(
        _scope: &Scope,
        values: &<Shared<[A]> as CubeType>::ExpandType,
    ) -> ErasedTensorExpand<T, ReadOnly> {
        ErasedTensorExpand::new(SmemPlainLoad::<A, T, W> {
            values: ExpandTypeClone::clone_unchecked(values),
            _t: PhantomData,
        })
    }
}

impl<T: Numeric> SmemPlainLoadSource<T> for ErasedTensor<T, ReadOnly> {}

/// A backing that replaces each cell it is handed, or adds into it, in a shared buffer read in
/// lines `W` scalars wide: a read and a write, not an atomic, so it declares [`WritesLines`]
/// alone. The rounds keep one writer at a cell at a time.
struct SmemCyclic<E: Numeric, W: Size> {
    values: <Shared<[E]> as CubeType>::ExpandType,
    first: NativeExpand<bool>,
    _e: PhantomData<(E, W)>,
}

impl<E: Numeric, W: Size> ErasedTensorOperationsExpand<E> for SmemCyclic<E, W> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        W::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        lines_of::expand::<E, W>(scope, &self.values)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        cyclic_line::expand::<E, W>(scope, &mut self.values, index, value.into(), self.first);
    }
}

impl<E: Numeric, W: Size> WritesLines<E> for SmemCyclic<E, W> {}

/// A backing that loads out of a shared buffer of plain `A` in lines `W` scalars wide, casting
/// to `T`.
struct SmemPlainLoad<A: Numeric, T: Numeric, W: Size> {
    values: <Shared<[A]> as CubeType>::ExpandType,
    _t: PhantomData<(T, W)>,
}

impl<A: Numeric, T: Numeric, W: Size> ErasedTensorOperationsExpand<T> for SmemPlainLoad<A, T, W> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        W::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        lines_of::expand::<A, W>(scope, &self.values)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        load_plain_line::expand::<A, T, W>(scope, &self.values, index).expand
    }
}

impl<A: Numeric, T: Numeric, W: Size> ReadsLines<T> for SmemPlainLoad<A, T, W> {}

/// The lines `W` scalars wide `values` holds.
#[cube]
fn lines_of<E: Numeric, W: Size>(values: &Shared<[E]>) -> usize {
    values.len() / W::value()
}

/// Land line `index`: each of its cells the value where `first`, the cell plus the value
/// otherwise.
#[cube]
fn cyclic_line<E: Numeric, W: Size>(
    values: &mut Shared<[E]>,
    index: usize,
    value: Vector<E, W>,
    first: bool,
) {
    #[unroll]
    for j in 0..W::value() {
        let cell = index * W::value() + j;
        if first {
            values[cell] = value.extract(j);
        } else {
            values[cell] += value.extract(j);
        }
    }
}

/// Load line `index`, cast to `T`.
#[cube]
fn load_plain_line<A: Numeric, T: Numeric, W: Size>(
    values: &Shared<[A]>,
    index: usize,
) -> Vector<T, W> {
    let mut line = Vector::<T, W>::empty();
    #[unroll]
    for j in 0..W::value() {
        line.insert(j, T::cast_from(values[index * W::value() + j]));
    }
    line
}
