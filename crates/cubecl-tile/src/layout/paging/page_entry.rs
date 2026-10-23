//! [`PageEntry`]: a page table entry a kernel hands to the cuts that resolve its page.

use cubecl::prelude::*;

use crate::*;

/// A page table entry and the cell it was read from: the page column `column` of sequence
/// `sequence` lists. A cut naming that cell takes `page` from it instead of loading the table.
#[derive(CubeType, CubeTypeMut, Clone)]
#[expand(derive(Clone))]
pub struct PageEntry {
    pub(crate) sequence: u32,
    pub(crate) column: u32,
    pub(crate) page: u32,
}

#[cube]
impl PageEntry {
    /// The entry no table lists, what a buffer that is not a pool of pages has. A cut that names
    /// it reads page 0, which is what the table gives outside itself.
    pub fn unlisted() -> PageEntry {
        PageEntry {
            sequence: 0xFFFF_FFFFu32,
            column: 0xFFFF_FFFFu32,
            page: 0u32,
        }
    }
}
