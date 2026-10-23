//! What a window cut over a pool of pages proves of the pages its lines lie in.

use crate::*;

/// What a window's extent proves of the pages its lines lie in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Spans {
    /// Nothing: a line leaving the pinned page reads its own off the table.
    Any,
    /// One sequence and at most a page of positions: the pinned page, or the next.
    TwoPages,
    /// One sequence inside the pinned page.
    OnePage,
}

impl Spans {
    /// What this proves of the same extent moved by a runtime amount, which loses the origin's
    /// alignment.
    pub(crate) fn shifted(self) -> Spans {
        match self {
            Spans::OnePage => Spans::TwoPages,
            spans => spans,
        }
    }
}

/// A window's extent along the paged axes, `None` where no level fixes it, and what its origin is
/// aligned to along the positions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct PagedWindow {
    pub(crate) sequences: Option<usize>,
    pub(crate) positions: Option<usize>,
    pub(crate) origin_alignment: usize,
}

impl PagedWindow {
    /// What this window proves over pages of `page_size` positions: one page where its positions
    /// fit in the alignment its origin and the pages share.
    pub(crate) fn spans(&self, page_size: usize) -> Spans {
        match (self.sequences, self.positions) {
            (Some(1), Some(positions)) if positions <= gcd(self.origin_alignment, page_size) => {
                Spans::OnePage
            }
            (Some(1), Some(positions)) if positions <= page_size => Spans::TwoPages,
            _ => Spans::Any,
        }
    }
}

/// What a window cut from the one a table is pinned for does with that pin.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Pin {
    /// The same pin: the window lies in the page this table pinned.
    Kept,
    /// Its own pin, for a window spanning pages as the variant says.
    Resolved(Spans),
}

/// How a window was cut from the one before, which decides what it keeps of that one's pages.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub(crate) enum WindowCut {
    /// The same origin and extent: handed to other units, or bounded short of its end.
    Unmoved,
    /// A level's tile, its extent along each dim in loads where the level fixes it; `nested` where
    /// the window is paged and every tile lies inside the window it was cut from.
    Level {
        edges: Vec<Option<usize>>,
        nested: bool,
    },
    /// Moved by a runtime amount: a `within` from a runtime start, or a gathered descent.
    Shifted,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(sequences: usize, positions: usize, origin_alignment: usize) -> PagedWindow {
        PagedWindow {
            sequences: Some(sequences),
            positions: Some(positions),
            origin_alignment,
        }
    }

    /// One page only where the alignment the origin and the pages share holds the whole extent:
    /// a step of 16 from 0 or a multiple of 16 stays in a page of 64; from a multiple of 4, or as
    /// 48 positions, it may cross into the next.
    #[test]
    fn a_window_is_inside_one_page_only_where_its_alignment_proves_it() {
        assert_eq!(window(1, 16, 0).spans(64), Spans::OnePage);
        assert_eq!(window(1, 16, 16).spans(64), Spans::OnePage);
        assert_eq!(window(1, 64, 128).spans(64), Spans::OnePage);
        assert_eq!(window(1, 16, 4).spans(64), Spans::TwoPages);
        assert_eq!(window(1, 48, 48).spans(64), Spans::TwoPages);
        assert_eq!(window(1, 128, 0).spans(64), Spans::Any);
        assert_eq!(window(2, 4, 0).spans(64), Spans::Any);
        let unfixed = PagedWindow {
            sequences: Some(1),
            positions: None,
            origin_alignment: 0,
        };
        assert_eq!(unfixed.spans(64), Spans::Any);
    }

    /// A runtime shift keeps the extent, so at most two pages, but loses the alignment.
    #[test]
    fn a_shifted_window_keeps_its_extent_not_its_page() {
        assert_eq!(Spans::OnePage.shifted(), Spans::TwoPages);
        assert_eq!(Spans::TwoPages.shifted(), Spans::TwoPages);
        assert_eq!(Spans::Any.shifted(), Spans::Any);
    }
}
