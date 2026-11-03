//! A shared-memory destination where each writer of a cube lands its partial in a slot of its own,
//! and a tile that reads the slots' sum back.
//!
//! Every writer replaces its own copy of the box at once, one cube barrier follows, and a read adds
//! the copies cell by cell, in writer order. No rounds, no atomics, no zeroing, and each cell takes
//! its partials in a fixed order, so the sum is the same bits from run to run. It holds a copy of
//! the box a writer: little where the box is, as a decode's output is, against the rounds a
//! [`SmemCyclicAccumulation`] synchronizes, which a box of fewer fragments than writers lands one
//! writer at a time.

use core::marker::PhantomData;

use cubecl::ir::{ExpandValue, VectorSize};
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl_std::tensor::{
    ErasedTensor, ErasedTensorExpand, ErasedTensorOperationsExpand, ReadsLines, WriteOnly,
    WritesLines,
};

use crate::*;

/// A cube's shared-memory accumulator over one box, summing in `A` and read back as `T`, with a
/// slot of its own for each of `writers` writers.
///
/// A writer drains its partial with [`drain`](SmemSlotsAccumulation::drain), which synchronizes
/// the cube once; once every writer has, [`source`] reads the sum. Every unit of the cube reaches
/// the drain, each plane with the partial it holds.
///
/// [`source`]: SmemSlotsAccumulation::source
#[derive(CubeType)]
pub struct SmemSlotsAccumulation<A: Numeric> {
    values: Shared<[A]>,
    /// The box the accumulator was opened over, which every writer's sink is shaped like.
    like: Tile,
    /// Reads the sum of the slots back at the opened tile's element; valid only once every
    /// writer has drained.
    pub source: Tile,
}

#[cube]
impl Tile {
    /// A shared-memory accumulator over this tile's box, summing in `A`, with a slot for each of
    /// `writers` writers. Nothing is zeroed: every writer replaces its whole slot. It is served in
    /// lines as wide as this tile's, so a partial drains into it and its sum lands in this tile
    /// line for line.
    pub fn smem_slots_accumulation<A: Numeric>(
        &self,
        #[comptime] writers: usize,
    ) -> SmemSlotsAccumulation<A> {
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
        let values = Shared::<[A]>::new_slice(comptime!(form.cells() * width * writers));
        // The previous open from this call site may still be read; let it finish first.
        sync_cube();

        let units = self.units();
        let source = Memory::smem_backed::<Served>(
            space,
            width,
            comptime!(FillUnits::cube(units)),
            Backing::new_ReadCall(RawCall::of::<Served>(
                ErasedTensor::<Served, ReadOnly>::of_smem_slots_sum::<A, W>(&values, writers),
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Replace),
            comptime!(Overhang::Never),
        );
        SmemSlotsAccumulation::<A> {
            values,
            like: self.clone(),
            source: source.nested_like(self),
        }
    }
}

#[cube]
impl<A: Numeric> SmemSlotsAccumulation<A> {
    /// The sink `writer` drains into: its own slot, each cell it is handed replaced.
    fn sink(&self, writer: usize) -> Tile {
        let space = comptime!(self.like.place.space.clone());
        let width = self.like.vector_size();
        let size!(W) = width;
        let form = comptime!(StageForm::dense(
            &space,
            width,
            StageStorage::Strided,
            LineBytes(A::size().comptime() * width)
        ));
        let slot = comptime!(form.cells() * width);
        let units = self.like.units();
        let sink = Memory::smem_backed::<A>(
            space,
            width,
            comptime!(FillUnits::cube(units)),
            Backing::new_WriteCall(RawCall::of::<A>(
                ErasedTensor::<A, WriteOnly>::of_smem_slot::<W>(&self.values, writer * slot, slot),
            )),
            comptime!(Packing::Plain),
            comptime!(form.clone()),
            RuntimeMap::integral(comptime!(form.physical_rank())),
            ComptimeOption::new_None(),
            comptime!(Write::Replace),
            comptime!(Overhang::Masked),
        );
        sink.nested_like(&self.like)
    }

    /// Drain `partial`, an accumulator over the whole box, into `writer`'s slot, the cube
    /// synchronized after. Once every writer of the cube has, the sum is in
    /// [`source`](SmemSlotsAccumulation::source).
    pub fn drain(&self, partial: &Tile, writer: usize) {
        partial.drained_into(&self.sink(writer));
        sync_cube();
    }

    /// Replace line `index` of the slot starting at `start` with `value`.
    #[allow(dead_code)] // Reached through its expand, from [`SmemSlot`].
    fn store_line<W: Size>(
        values: &mut Shared<[A]>,
        start: usize,
        index: usize,
        value: Vector<A, W>,
    ) {
        #[unroll]
        for j in 0..W::value() {
            values[start + index * W::value() + j] = value.extract(j);
        }
    }

    /// Line `index` summed over the `writers` slots, in writer order, cast to `T`.
    #[allow(dead_code)] // Reached through its expand, from [`SmemSlotsSum`].
    fn sum_line<T: Numeric, W: Size>(
        values: &Shared<[A]>,
        #[comptime] writers: usize,
        index: usize,
    ) -> Vector<T, W> {
        let slot = values.len() / writers;
        let mut line = Vector::<T, W>::empty();
        #[unroll]
        for j in 0..W::value() {
            let cell = index * W::value() + j;
            let mut sum = values[cell];
            #[unroll]
            for writer in 1..writers {
                sum += values[writer * slot + cell];
            }
            line.insert(j, T::cast_from(sum));
        }
        line
    }
}

/// The erased tensors over a shared buffer of slots.
pub(crate) trait SmemSlotsSink<E: Numeric> {
    /// The sink that replaces cells of the slot `len` scalars long starting at `start` in
    /// `values`, in lines `W` scalars wide.
    fn of_smem_slot<W: Size>(
        _values: &Shared<[E]>,
        _start: usize,
        _len: usize,
    ) -> ErasedTensor<E, WriteOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_slot<W: Size>(
        _scope: &Scope,
        values: &<Shared<[E]> as CubeType>::ExpandType,
        start: NativeExpand<usize>,
        len: usize,
    ) -> ErasedTensorExpand<E, WriteOnly> {
        ErasedTensorExpand::new(SmemSlot::<E, W> {
            values: ExpandTypeClone::clone_unchecked(values),
            start,
            len,
            _e: PhantomData,
        })
    }
}

impl<E: Numeric> SmemSlotsSink<E> for ErasedTensor<E, WriteOnly> {}

/// The source that sums the slots of a shared buffer of plain `A`, served as `T`.
pub(crate) trait SmemSlotsSumSource<T: Numeric> {
    /// The source that sums the `writers` slots of `values`, cast to `T`, in lines `W` scalars
    /// wide.
    fn of_smem_slots_sum<A: Numeric, W: Size>(
        _values: &Shared<[A]>,
        _writers: usize,
    ) -> ErasedTensor<T, ReadOnly> {
        unexpanded!()
    }

    fn __expand_of_smem_slots_sum<A: Numeric, W: Size>(
        _scope: &Scope,
        values: &<Shared<[A]> as CubeType>::ExpandType,
        writers: usize,
    ) -> ErasedTensorExpand<T, ReadOnly> {
        ErasedTensorExpand::new(SmemSlotsSum::<A, T, W> {
            values: ExpandTypeClone::clone_unchecked(values),
            writers,
            _t: PhantomData,
        })
    }
}

impl<T: Numeric> SmemSlotsSumSource<T> for ErasedTensor<T, ReadOnly> {}

/// A backing that replaces cells of one slot of a shared buffer, in lines `W` scalars wide.
struct SmemSlot<E: Numeric, W: Size> {
    values: <Shared<[E]> as CubeType>::ExpandType,
    start: NativeExpand<usize>,
    /// Scalars a slot holds.
    len: usize,
    _e: PhantomData<(E, W)>,
}

impl<E: Numeric, W: Size> ErasedTensorOperationsExpand<E> for SmemSlot<E, W> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        W::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        (self.len / W::__expand_value(scope)).__expand_runtime_method(scope)
    }

    fn __expand_write_line_method(
        &mut self,
        scope: &Scope,
        index: NativeExpand<usize>,
        value: ExpandValue,
    ) {
        SmemSlotsAccumulation::<E>::__expand_store_line::<W>(
            scope,
            &mut self.values,
            self.start,
            index,
            value.into(),
        );
    }
}

impl<E: Numeric, W: Size> WritesLines<E> for SmemSlot<E, W> {}

/// A backing that sums the slots of a shared buffer of plain `A` in lines `W` scalars wide,
/// casting to `T`.
struct SmemSlotsSum<A: Numeric, T: Numeric, W: Size> {
    values: <Shared<[A]> as CubeType>::ExpandType,
    writers: usize,
    _t: PhantomData<(T, W)>,
}

impl<A: Numeric, T: Numeric, W: Size> ErasedTensorOperationsExpand<T> for SmemSlotsSum<A, T, W> {
    fn __expand_vector_size_method(&self, scope: &Scope) -> VectorSize {
        W::__expand_value(scope)
    }

    fn __expand_lines_method(&self, scope: &Scope) -> NativeExpand<usize> {
        let scalars = self.values.__expand_len_method(scope);
        let per_line = (self.writers * W::__expand_value(scope)).__expand_runtime_method(scope);
        scalars.__expand_div_method(scope, per_line)
    }

    fn __expand_read_line_method(&self, scope: &Scope, index: NativeExpand<usize>) -> ExpandValue {
        SmemSlotsAccumulation::<A>::__expand_sum_line::<T, W>(
            scope,
            &self.values,
            self.writers,
            index,
        )
        .expand
    }
}

impl<A: Numeric, T: Numeric, W: Size> ReadsLines<T> for SmemSlotsSum<A, T, W> {}
