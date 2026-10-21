//! The tensor-core encoding of a plane tile ([`CmmaData`]) and its fragment↔memory transports.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    ir::{ElemType, FloatKind},
    prelude::*,
};

use crate::*;

/// A tensor-core fragment plus the comptime config its load/store paths dispatch on.
/// `Clone` duplicates the handle, not the fragment.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub(crate) struct CmmaData {
    /// The fragment, kept at [`Held`] and named at the element it holds where it is used.
    pub fragment: Matrix<Held>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
    /// The whole MMA tile's `(m, n)`, whatever the role.
    #[cube(comptime)]
    pub shape: (usize, usize),
    /// This plane's one-tile window of shared memory the fragment bounces through, at the fragment's
    /// element.
    pub scratch: ComptimeOption<Raw<SharedSlice>>,
}

#[cube]
impl CmmaData {
    /// Allocate an uninitialized fragment. `m`/`n`/`k` are the whole MMA tile, whatever the role.
    pub(crate) fn alloc<T: Numeric>(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
    ) -> CmmaData {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, layout) };
        CmmaData {
            fragment: unsafe { matrix.downcast_unchecked::<Held>() },
            ident,
            layout,
            shape: comptime!((m, n)),
            scratch: ComptimeOption::new_None(),
        }
    }

    /// This fragment carrying `scratch`, one tile of shared memory for its plane.
    pub(crate) fn with_scratch<T: Numeric>(self, scratch: Shared<[T]>) -> CmmaData {
        CmmaData {
            fragment: self.fragment,
            ident: comptime!(self.ident),
            layout: comptime!(self.layout),
            shape: comptime!(self.shape),
            scratch: ComptimeOption::new_Some(Raw::<SharedSlice>::of::<T>(scratch)),
        }
    }

    /// Store this tile row-major into `scratch`, one tile's cells from its start.
    pub(crate) fn store_scratch<T: Numeric>(&self, scratch: &Shared<[T]>) {
        let n = comptime!(self.shape.1 as u32);
        let mut window = scratch.clone();
        cmma::store(
            &mut window,
            &self.fragment.downcast::<T>(),
            n,
            MatrixLayout::RowMajor,
        )
    }

    /// Load this tile back from `scratch`.
    pub(crate) fn load_scratch<T: Numeric>(&mut self, scratch: &Shared<[T]>) {
        let n = comptime!(self.shape.1 as u32);
        cmma::load_with_layout(
            &mut self.fragment.downcast::<T>(),
            scratch,
            n,
            MatrixLayout::RowMajor,
        )
    }

    /// Zero the fragment.
    pub(crate) fn zero<T: Numeric>(&mut self) {
        cmma::fill(&mut self.fragment.downcast::<T>(), T::from_int(0));
    }

    /// Fill this fragment from `mem`'s window, stepping rows by the physical row stride.
    pub(crate) fn load_window<T: Numeric>(&mut self, mem: &Memory, #[comptime] row: usize) {
        let element = mem.stage_element();
        comptime!(assert!(
            element.holds_values(),
            "CmmaData::load_window: a cmma fragment loads at one element type, so it cannot \
             unpack a packed source as it reads; contract it, which lands it, or decode it into a \
             stage (`stage.copy_from(&w.mul(&scales))`)"
        ));
        let stride = mem.row_stride_at(row);
        match comptime!(self.ident) {
            MatrixIdent::Accumulator => cmma::load_with_layout(
                &mut self.fragment.downcast::<T>(),
                mem.window_slice::<T>(),
                stride,
                comptime!(self.layout),
            ),
            _ => cmma::load(
                &mut self.fragment.downcast::<T>(),
                mem.window_slice::<T>(),
                stride,
            ),
        }
    }

    /// Drain this fragment into `mem`'s *window* (origin offset, physical row stride).
    pub(crate) fn store_window<T: Numeric>(&self, mem: &mut Memory, #[comptime] row: usize) {
        let stride = mem.row_stride_at(row);
        cmma::store(
            mem.window_slice_mut::<T>(),
            &self.fragment.downcast::<T>(),
            stride,
            comptime!(self.layout),
        )
    }

    /// Every cell multiplied by `factor`, bounced through the fragment's slot of the plane's
    /// scratch as [`mul_along`](Self::mul_along) bounces it: a cmma fragment is not read cell by
    /// cell.
    pub(crate) fn scale<T: Numeric>(&self, factor: T) {
        self.spill_to_scratch::<T>();
        sync_plane();
        let mut scratch = self.scratch_slot::<T>();
        let (m, n) = self.shape;
        let mut cell = UNIT_POS_PLANE as usize;
        while cell < m * n {
            scratch[cell] *= factor;
            cell += PLANE_DIM as usize;
        }
        sync_plane();
        self.reload_from_scratch::<T>();
        sync_plane();
    }

    /// `self[i, :] *= factors[first + i]` for every slice `i` along the fragment's columns,
    /// bounced through its slot of the plane's scratch: spilled, scaled by the plane's units in
    /// turns, and reloaded, each step met on `sync_plane`. The slot is the plane's own, so no
    /// other plane waits; every unit of the plane calls it.
    pub(crate) fn mul_along<T: Numeric>(&self, factors: &Array<T>, #[comptime] first: usize) {
        self.spill_to_scratch::<T>();
        sync_plane();
        self.mul_spilled(factors, first);
        sync_plane();
        self.reload_from_scratch::<T>();
        sync_plane();
    }

    /// `slot[i, :] *= factors[first + i]` over this fragment's spilled cells, the plane's units
    /// taking them in turns. The caller owns the barriers.
    pub(crate) fn mul_spilled<T: Numeric>(&self, factors: &Array<T>, #[comptime] first: usize) {
        let mut scratch = self.scratch_slot::<T>();
        let (m, n) = self.shape;
        let mut cell = UNIT_POS_PLANE as usize;
        while cell < m * n {
            scratch[cell] *= factors[first + cell / n];
            cell += PLANE_DIM as usize;
        }
    }

    /// Load this fragment back from its slot of the plane's scratch. The caller owns the barriers.
    pub(crate) fn reload_from_scratch<T: Numeric>(&self) {
        let scratch = self.scratch_slot::<T>();
        let mut fragment = self.clone();
        fragment.load_scratch(&scratch);
    }

    /// Store this fragment into its slot of the plane's scratch. The caller owns the barriers.
    pub(crate) fn spill_to_scratch<T: Numeric>(&self) {
        self.store_scratch(&self.scratch_slot::<T>());
    }

    /// Write this fragment's spilled cells into `mem` through the store's own write.
    pub(crate) fn add_from_scratch<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: Space,
    ) {
        let scratch = self.scratch_slot::<T>();
        let (m, n) = comptime!(self.shape);
        let width = comptime!(mem.store.vector_size);
        comptime!(assert!(
            n.is_multiple_of(width),
            "CmmaData::add_from_scratch: the store's lines ({width}) do not divide the \
             fragment's columns ({n})"
        ));
        let size!(W) = width;
        let lines_per_row = comptime!(n / width);
        let lines = comptime!(m * lines_per_row);
        let axes = comptime!(MatrixAxes::trailing(&space));
        let mut sink = mem.matrix_mut::<Out, W>(0usize, axes, space);
        let plane_units = PLANE_DIM as usize;
        let mut line = UNIT_POS_PLANE as usize;
        while line < lines {
            let mut value = Vector::<Out, W>::empty();
            #[unroll]
            for e in 0..width {
                value.insert(e, Out::cast_from(scratch[line * width + e]));
            }
            sink.write(
                ((line / lines_per_row) as u32, (line % lines_per_row) as u32),
                value,
            );
            line += plane_units;
        }
    }

    /// Drain this fragment through the plane's scratch, with cube-wide syncs.
    pub(crate) fn bounce_cast_window<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] space: Space,
    ) {
        sync_cube();
        self.spill_to_scratch::<T>();
        sync_cube();
        self.add_from_scratch::<T, Out>(mem, space);
        sync_cube();
    }

    /// This fragment's slot of the plane's scratch, or the reason there is none.
    fn scratch_slot<T: Numeric>(&self) -> Shared<[T]> {
        #[comptime]
        match &self.scratch {
            ComptimeOption::Some(scratch) => scratch.at::<T>().clone(),
            ComptimeOption::None => panic!(
                "CmmaData: a fragment bounces through a scratch, to scale its slices, to reach a \
                 store that folds, or a window the problem's edge cuts short; open the \
                 accumulator with `with_scratch`"
            ),
        }
    }

    /// Drain this fragment into `mem`'s window, casting `T` down to the sink's element type.
    pub(crate) fn store_cast_window<T: Numeric, Out: Numeric>(
        &self,
        mem: &mut Memory,
        #[comptime] row: usize,
    ) {
        let stride = mem.row_stride_at(row);
        let casted: Matrix<Out> = cmma::cast(&self.fragment.downcast::<T>());
        cmma::store(
            mem.window_slice_mut::<Out>(),
            &casted,
            stride,
            comptime!(self.layout),
        )
    }
}

/// Whether a fragment of `T` cast to `Out` is a fragment the device holds at `(m, n)`: the same
/// element, or one an accumulator is offered in at that shape. `wmma` offers none in `bf16`, so a
/// fragment cast there is a kernel that fails to compile, and a `bf16` sink takes the cells through
/// the scratch instead, each cast on its write. A device whose properties are not known at
/// expansion is taken at its word.
// `T` and `Out` are read by the expansion, which is where a type has its element.
#[allow(clippy::extra_unused_type_parameters)]
#[cube]
pub(crate) fn casts_in_place<T: Numeric, Out: Numeric>(
    #[comptime] m: usize,
    #[comptime] n: usize,
) -> comptime_type!(bool) {
    intrinsic!(|scope| {
        let (from, to) = (T::elem_type(scope), Out::elem_type(scope));
        from == to
            || scope
                .state()
                .device_properties
                .as_ref()
                .is_none_or(|properties| {
                    cmma_cast_layout_is_compatible(&properties.identity.fingerprint, from, to)
                        && properties.features.matmul.cmma.iter().any(|offered| {
                            offered.cd_type == to
                                && offered.m as usize == m
                                && offered.n as usize == n
                        })
                })
    })
}

/// Volta uses different register layouts for f32 and f16 accumulator fragments, so cross-type
/// casts must bounce through shared memory.
fn cmma_cast_layout_is_compatible(fingerprint: &str, from: ElemType, to: ElemType) -> bool {
    let f32 = ElemType::Float(FloatKind::F32);
    let f16 = ElemType::Float(FloatKind::F16);
    !(fingerprint == "ptx_sm70" && ((from == f32 && to == f16) || (from == f16 && to == f32)))
}

#[cfg(test)]
mod accumulator_cast_layout_tests {
    use super::*;

    #[test]
    fn volta_bounces_f32_f16_accumulator_casts() {
        let (f32, f16) = (
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F16),
        );
        assert!(!cmma_cast_layout_is_compatible("ptx_sm70", f32, f16));
        assert!(!cmma_cast_layout_is_compatible("ptx_sm70", f16, f32));
    }

    #[test]
    fn later_architectures_keep_the_in_place_cast_path() {
        let (f32, f16) = (
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F16),
        );
        assert!(cmma_cast_layout_is_compatible("ptx_sm75", f32, f16));
        assert!(cmma_cast_layout_is_compatible("ptx_sm80", f32, f16));
    }

    #[test]
    fn same_type_cast_does_not_depend_on_architecture() {
        let f16 = ElemType::Float(FloatKind::F16);
        assert!(cmma_cast_layout_is_compatible("ptx_sm70", f16, f16));
    }
}

/// How a cmma fragment reaches its destination.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum FragmentDrain {
    /// The fragment's store intrinsic: a replacing destination, window wholly inside it.
    Intrinsic,
    /// Through the plane's scratch: a destination that adds, or a window cut short by the edge.
    Bounce,
}

impl FragmentDrain {
    /// How a fragment drains into a destination written as `access` says, `addressed` when its
    /// values sit at an address the intrinsic can store to, `in_place` when the fragment cast to
    /// the destination's element is one the device holds ([`casts_in_place`]).
    pub(crate) const fn of(access: &Access, addressed: bool, in_place: bool) -> Self {
        match (access.write, access.overhang, addressed && in_place) {
            (Write::Replace, Overhang::Never | Overhang::Fits, true) => FragmentDrain::Intrinsic,
            (Write::Replace, _, _) | (Write::Accumulate | Write::Exclusive(_), _, _) => {
                FragmentDrain::Bounce
            }
        }
    }
}

#[cfg(test)]
mod fragment_drain_tests {
    use super::*;

    fn access(write: Write, overhang: Overhang) -> Access {
        Access {
            whole: false,
            overhang,
            write,
            fill: FillUnits::cube(0),
            storage: WindowStorage::Stored(Storage::Strided),
            delivery: Delivery::SyncPerUnit,
        }
    }

    #[test]
    fn a_replacing_window_inside_its_buffer_stores_through_the_intrinsic() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            let drain = FragmentDrain::of(&access(Write::Replace, overhang), true, true);
            assert_eq!(drain, FragmentDrain::Intrinsic);
        }
    }

    #[test]
    fn an_overhanging_or_folding_window_bounces() {
        let masked = FragmentDrain::of(&access(Write::Replace, Overhang::Masked), true, true);
        assert_eq!(masked, FragmentDrain::Bounce);
        for overhang in [Overhang::Never, Overhang::Fits, Overhang::Masked] {
            for write in [Write::Accumulate, Write::Exclusive(Schedule::Sequential)] {
                assert_eq!(
                    FragmentDrain::of(&access(write, overhang), true, true),
                    FragmentDrain::Bounce,
                    "{write:?} {overhang:?}"
                );
            }
        }
    }

    /// A fragment cast to an element the device holds no fragment of cannot be stored whole, so
    /// even a replacing window wholly inside its buffer bounces, its cells cast on their write.
    #[test]
    fn a_fragment_with_no_cast_in_place_bounces() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            let drain = FragmentDrain::of(&access(Write::Replace, overhang), true, false);
            assert_eq!(drain, FragmentDrain::Bounce);
        }
    }

    /// A write call has no address for the intrinsic to store to, so even a replacing window
    /// wholly inside it bounces.
    #[test]
    fn a_window_with_no_address_bounces() {
        for overhang in [Overhang::Never, Overhang::Fits] {
            let drain = FragmentDrain::of(&access(Write::Replace, overhang), false, true);
            assert_eq!(drain, FragmentDrain::Bounce);
        }
    }
}

#[cfg(test)]
mod held_fragment_tests {
    use cubecl::ir::{AddressType, Scope, settings};
    use half::f16;

    use super::*;

    fn test_scope() -> Scope {
        Scope::root(settings::KernelSettings::new(
            settings::Dim3::new_single(),
            settings::ExecutionMode::Checked,
            AddressType::U32,
        ))
    }

    /// A `16 × 16` accumulator fragment allocated at `H`, kept at [`Held`], and stored into a
    /// scratch of `N`.
    #[cube]
    fn fragment_stored_at<H: Numeric, N: Numeric>() {
        let data = CmmaData::alloc::<H>(
            MatrixIdent::Accumulator,
            16usize,
            16usize,
            16usize,
            MatrixLayout::Undefined,
        );
        let scratch = Shared::<[N]>::new_slice(256usize);
        data.store_scratch::<N>(&scratch);
    }

    /// A fragment kept at [`Held`] is named again at the element it was allocated at.
    #[test]
    fn a_held_fragment_is_named_at_its_element() {
        fragment_stored_at::expand::<f16, f16>(&test_scope());
    }

    /// A fragment kept at [`Held`] and named at an element it does not hold is refused at
    /// expansion.
    ///
    /// The tile keeps every fragment at one dummy type, so nothing in Rust ties a fragment to the
    /// element it was allocated at: only this check stops an `f16` fragment being stored as an
    /// `f32` one.
    #[test]
    #[should_panic(expected = "Matrix::downcast: the fragment holds")]
    fn a_held_fragment_refuses_another_element() {
        fragment_stored_at::expand::<f16, f32>(&test_scope());
    }
}
