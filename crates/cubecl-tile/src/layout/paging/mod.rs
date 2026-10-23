//! A pool of fixed-size pages read as sequences: [`Paging`] states it, [`PageTable`] resolves it.
//!
//! The buffer is a pool whose `sequence` dim holds pages and whose `position` dim a page's
//! positions. An operand bound over it is read as the sequences a table names: sequence `b`'s
//! position `s` lies in page `table[b, s / page_size]`, at row `s % page_size`. The projection stays
//! direct over the logical `[…, sequence, …, position, …]` axes, so every window, walk and bound
//! is the logical one; the table is resolved only where a coordinate becomes an offset
//! ([`BufferLayout`](super::projection::BufferLayout)), at every offset taken, a masked cell's
//! included. The pool is bound as it lies; the operand's extents along the two axes are the
//! table's, read off it at launch, so nothing states them twice. A coordinate past the table reads
//! the pool's first page, and a table entry past the pool its last: a table is device data a
//! launch may never have written.
//!
//! The table is read as each window is cut, not as each line is: a window of one sequence pins the
//! pages its positions may reach, at most `MAX_PINNED_PAGES`, its lines offset from those pages'
//! starts without touching the table. A window proved inside one page pins it alone, and a window
//! cut inside that one keeps the pin. A wider window resolves each line.

pub(crate) mod base;
pub(crate) mod page_entry;
pub(crate) mod page_table;
pub(crate) mod window;

pub use base::Paging;
pub use page_entry::PageEntry;
pub(crate) use page_table::PageTable;
pub(crate) use window::WindowCut;
