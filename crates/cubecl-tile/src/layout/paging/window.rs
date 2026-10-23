//! What a window cut over a pool of pages proves of the pages its lines lie in.

use crate::*;

/// The most pages a window pins: each is a register its lines pick from and a table read at its
/// cut. A window reaching more resolves its lines one by one.
pub(crate) const MAX_PINNED_PAGES: usize = 8;

/// What a window's extent proves of the pages its lines lie in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Spans {
    /// Nothing: a line leaving the pinned page reads its own off the table.
    Any,
    /// One sequence over the pinned page and at most `pages - 1` after it; `unaligned` pages from
    /// an origin of unknown alignment, as once the window is moved by a runtime amount.
    Pages { pages: usize, unaligned: usize },
}

impl Spans {
    /// What this proves of the same extent moved by a runtime amount, which loses the origin's
    /// alignment.
    pub(crate) fn shifted(self) -> Spans {
        match self {
            Spans::Pages { unaligned, .. } if unaligned <= MAX_PINNED_PAGES => Spans::Pages {
                pages: unaligned,
                unaligned,
            },
            _ => Spans::Any,
        }
    }

    /// The pages a window pins: none where its lines resolve their own.
    pub(crate) fn pages(self) -> usize {
        match self {
            Spans::Any => 0,
            Spans::Pages { pages, .. } => pages,
        }
    }
}

/// The pages `positions` consecutive positions may reach from an origin aligned to `alignment`,
/// which divides `page_size`.
fn pages_reached(positions: usize, alignment: usize, page_size: usize) -> usize {
    (page_size - alignment + positions.max(1) - 1) / page_size + 1
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
    /// What this window proves over pages of `page_size` positions: the pages its positions may
    /// reach from an origin aligned as the origin and the pages share.
    pub(crate) fn spans(&self, page_size: usize) -> Spans {
        match (self.sequences, self.positions) {
            (Some(1), Some(positions)) => {
                let alignment = gcd(self.origin_alignment, page_size);
                match pages_reached(positions, alignment, page_size) {
                    pages if pages <= MAX_PINNED_PAGES => Spans::Pages {
                        pages,
                        unaligned: pages_reached(positions, 1, page_size),
                    },
                    _ => Spans::Any,
                }
            }
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

    fn pages(pages: usize, unaligned: usize) -> Spans {
        Spans::Pages { pages, unaligned }
    }

    /// One page only where the alignment the origin and the pages share holds the whole extent:
    /// a step of 16 from 0 or a multiple of 16 stays in a page of 64; from a multiple of 4, or as
    /// 48 positions, it may cross into the next.
    #[test]
    fn a_window_is_inside_one_page_only_where_its_alignment_proves_it() {
        assert_eq!(window(1, 16, 0).spans(64), pages(1, 2));
        assert_eq!(window(1, 16, 16).spans(64), pages(1, 2));
        assert_eq!(window(1, 64, 128).spans(64), pages(1, 2));
        assert_eq!(window(1, 16, 4).spans(64), pages(2, 2));
        assert_eq!(window(1, 48, 48).spans(64), pages(2, 2));
        assert_eq!(window(2, 4, 0).spans(64), Spans::Any);
        let unfixed = PagedWindow {
            sequences: Some(1),
            positions: None,
            origin_alignment: 0,
        };
        assert_eq!(unfixed.spans(64), Spans::Any);
    }

    /// A window wider than a page pins every page it may reach: 128 positions from a page's start
    /// reach two pages of 64, from anywhere three; past the cap it resolves line by line.
    #[test]
    fn a_window_wider_than_a_page_pins_the_pages_it_reaches() {
        assert_eq!(window(1, 128, 0).spans(64), pages(2, 3));
        assert_eq!(window(1, 128, 0).spans(16), pages(8, 9));
        assert_eq!(window(1, 128, 8).spans(16), Spans::Any);
        assert_eq!(window(1, 256, 0).spans(16), Spans::Any);
    }

    /// A runtime shift keeps the extent but loses the alignment, so the pages any origin reaches.
    #[test]
    fn a_shifted_window_keeps_its_extent_not_its_page() {
        assert_eq!(pages(1, 2).shifted(), pages(2, 2));
        assert_eq!(pages(2, 2).shifted(), pages(2, 2));
        assert_eq!(pages(8, 9).shifted(), Spans::Any);
        assert_eq!(Spans::Any.shifted(), Spans::Any);
    }
}
