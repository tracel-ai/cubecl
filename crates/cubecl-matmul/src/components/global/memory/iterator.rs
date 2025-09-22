use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};

#[derive(Clone, CubeType)]
/// An iterator over global memory, advancing along k.
pub struct GlobalIterator<EI: Numeric> {
    global_view: View<Line<EI>, Coords2d>,
    offset: RuntimeCell<u32>,
    /// The amount to advance by on each iteration
    step: u32,
    view_size: Coords2d,
    #[cube(comptime)]
    view_direction: ViewDirection,
    #[cube(comptime)]
    checked: bool,
}

unsafe impl<EG: Numeric> Sync for GlobalIterator<EG> {}
unsafe impl<EG: Numeric> Send for GlobalIterator<EG> {}

#[derive(CubeType, Clone, Copy)]
pub enum ViewDirection {
    Row,
    Col,
    /// Cannot advance if direction is none
    None,
}

#[cube]
impl<EG: Numeric> GlobalIterator<EG> {
    /// Instantiate a read iterator over the given global view, which should be sliced to the size
    /// of one `m`/`n` stage and the full range of `k` handled by this matmul instance.
    ///
    /// `step` is the amount advanced in `view_direction` each iteration.
    /// `checked` determines whether the slices should be created as checked or unchecked.
    pub fn new(
        global_view: View<Line<EG>, Coords2d>,
        step: u32,
        #[comptime] view_direction: ViewDirection,
        #[comptime] checked: bool,
    ) -> Self {
        let (size_row, size_col) = global_view.shape();
        let view_size = match view_direction {
            ViewDirection::Row => (step, size_col),
            ViewDirection::Col => (size_row, step),
            ViewDirection::None => (size_row, size_col),
        };

        GlobalIterator::<EG> {
            global_view,
            offset: RuntimeCell::new(0),
            step,
            view_size,
            view_direction,
            checked,
        }
    }

    /// Advance the view along the k dimension by a specified offset, `k_offset`.
    pub fn advance(&self) {
        self.offset.store(self.offset.read() + self.step);
    }

    /// Returns the current view slice of the iterator
    pub fn view(&self) -> View<Line<EG>, Coords2d> {
        let offset = match comptime![self.view_direction] {
            ViewDirection::Row => (self.offset.read(), 0u32),
            ViewDirection::Col => (0u32, self.offset.read()),
            ViewDirection::None => (0u32, 0u32).runtime(),
        };
        if comptime![self.checked] {
            self.global_view.slice(offset, self.view_size)
        } else {
            self.global_view.slice_unchecked(offset, self.view_size)
        }
    }
}
