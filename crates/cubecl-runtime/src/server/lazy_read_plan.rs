use crate::server::{BufferBinding, CopyDescriptor, Handle};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_zspace::striding::try_check_contiguous_row_major_strides;

/// Where the bytes a lazy read still holds lie in its memory, and the copies that read them.
///
/// A read narrower than its whole is held only through a view, which only a contiguous source
/// gives, so a strided source is held whole or not at all.
pub(crate) struct LazyReadPlan<'a> {
    /// What the read reads.
    pub source: &'a CopyDescriptor<Handle>,
    /// The bytes of the read still held, merged, in byte order.
    pub held: Vec<Range<usize>>,
}

impl LazyReadPlan<'_> {
    /// Whether `written`, of the read's memory, shares a byte with what is held.
    pub(crate) fn is_overwritten_by(&self, written: &BufferBinding) -> bool {
        self.pieces().iter().any(|piece| written.overlaps(piece))
    }

    /// The copies of what is held, one per piece, with the offset of each in the read.
    pub(crate) fn to_sources(&self) -> (Vec<CopyDescriptor>, Vec<usize>) {
        match (self.is_contiguous(), self.held.is_empty()) {
            (true, _) => (
                self.held.iter().map(|range| self.narrowed(range)).collect(),
                self.held.iter().map(|range| range.start).collect(),
            ),
            (false, true) => (Vec::new(), Vec::new()),
            (false, false) => (vec![self.source.clone().binding()], vec![0]),
        }
    }

    /// The memory what is held lies in.
    fn pieces(&self) -> Vec<Range<u64>> {
        let read = self.source.handle.to_binding().range();
        match (self.is_contiguous(), self.held.is_empty()) {
            (true, _) => self
                .held
                .iter()
                .map(|range| read.start + range.start as u64..read.start + range.end as u64)
                .collect(),
            (false, true) => Vec::new(),
            (false, false) => vec![read],
        }
    }

    /// Whether its bytes are one run of memory, a scalar's included.
    fn is_contiguous(&self) -> bool {
        self.source.shape.is_empty()
            || try_check_contiguous_row_major_strides(&self.source.shape, &self.source.strides)
                .is_ok()
    }

    /// A copy of the bytes `range` of the read, as bytes.
    fn narrowed(&self, range: &Range<usize>) -> CopyDescriptor {
        let mut handle = self.source.clone().binding().handle;
        let base = handle.offset_start.unwrap_or(0);
        handle.offset_end = Some(handle.size - (base + range.end as u64));
        handle.offset_start = Some(base + range.start as u64);
        CopyDescriptor::new(handle, [range.len()].into(), [1].into(), 1)
    }
}
