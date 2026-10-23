use cubecl::prelude::*;
use cubecl_std::tensor::{View, ViewMut, layout::Coordinates};

/// A [`View`] plus a comptime `check` flag: reads past the overhang zero, writes skip.
#[derive(CubeType)]
pub struct Masked<'a, T: CubePrimitive, C: Coordinates + 'a> {
    view: View<'a, T, C>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates + 'a> Masked<'a, T, C> {
    pub fn new(view: View<'a, T, C>, #[comptime] check: bool) -> Self {
        Masked::<'a, T, C> { view, check }
    }

    pub fn read(&self, pos: C) -> T {
        if comptime!(self.check) {
            self.view.read_checked(pos)
        } else {
            // `check == false`: the launch proved this access in-bounds.
            self.view.read_unchecked(pos)
        }
    }

    /// The value at `pos`, untested: the reader proved it in bounds, so reading past them is out
    /// of bounds, not masked.
    pub fn read_unchecked(&self, pos: C) -> T {
        self.view.read_unchecked(pos)
    }

    /// Whether `pos` lands on real data; always `true` when `check` is `false`.
    pub fn is_in_bounds(&self, pos: C) -> bool {
        if comptime!(self.check) {
            self.view.is_in_bounds(pos)
        } else {
            true.runtime()
        }
    }

    /// Whether the non-empty box at `pos` with `extent` is wholly in bounds.
    pub(crate) fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        if comptime!(self.check) {
            let one = C::from_int(pos.clone(), 1i64);
            let far = C::sub(C::add(pos, extent), one);
            self.view.is_in_bounds(far)
        } else {
            true.runtime()
        }
    }

    pub fn shape(&self) -> C {
        self.view.shape()
    }
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates + 'static> Masked<'a, T, C> {
    /// The buffer the stage's arrangement places at `pos`, spanning `size`: the address `ldmatrix`
    /// reads. Panics where the view masks.
    pub(crate) fn line_slice(&self, pos: C, size: C) -> &[T] {
        comptime!(assert!(
            !self.check,
            "Masked::line_slice: an address reads past the overhang a masked read would zero; \
             serve the operand from a stage whose accesses the launch proved in bounds"
        ));
        self.view
            .clone()
            .slice_unchecked(pos, size)
            .as_linear_slice()
    }

    /// The one item at `pos` as a run of the buffer, what an async copy reads by address: empty
    /// where the view masks and `pos` falls past the bound, so the copy reads nothing and the
    /// engine zero-fills the item a masked read would have zeroed.
    pub(crate) fn item_run(&self, pos: C) -> &[T] {
        let one = C::from_int(pos.clone(), 1i64);
        let run = self
            .view
            .clone()
            .slice_unchecked(pos.clone(), one)
            .as_linear_slice();
        if comptime!(self.check) {
            let len = select(self.view.is_in_bounds(pos), 1usize, 0usize);
            &run[0..len]
        } else {
            run
        }
    }
}

/// The mutable twin of [`Masked`].
#[derive(CubeType)]
pub struct MaskedMut<'a, T: CubePrimitive, C: Coordinates + 'a> {
    view: ViewMut<'a, T, C>,
    #[cube(comptime)]
    pub(crate) check: bool,
}

#[cube]
impl<'a, T: CubePrimitive, C: Coordinates + 'a> MaskedMut<'a, T, C> {
    pub fn new(view: ViewMut<'a, T, C>, #[comptime] check: bool) -> Self {
        MaskedMut::<'a, T, C> { view, check }
    }

    pub fn read(&self, pos: C) -> T {
        if comptime!(self.check) {
            self.view.read_checked(pos)
        } else {
            self.view.read_unchecked(pos)
        }
    }

    pub fn write(&mut self, pos: C, value: T) {
        if comptime!(self.check) {
            self.view.write_checked(pos, value);
        } else {
            self.view.write(pos, value);
        }
    }

    /// Mutable counterpart to [`Masked::block_in_bounds`].
    pub(crate) fn block_in_bounds(&self, pos: C, extent: C) -> bool {
        if comptime!(self.check) {
            let one = C::from_int(pos.clone(), 1i64);
            let far = C::sub(C::add(pos, extent), one);
            self.view.is_in_bounds(far)
        } else {
            true.runtime()
        }
    }

    pub fn shape(&self) -> C {
        self.view.shape()
    }
}
