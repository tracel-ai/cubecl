//! The scanned transport: source and destination each read as a flat run of its own window.

use cubecl::prelude::*;

use super::base::Scan;
use super::straight::fill_extent;
use crate::*;

#[cube]
impl Memory {
    /// General path of [`fill_from`](Memory::fill_from); `scan` is what the source hands out.
    pub(crate) fn fill_scanned<T: Numeric, W: Size>(
        &mut self,
        src: &Memory,
        #[comptime] scan: Scan,
    ) {
        match scan {
            Scan::Element => self.scan_unpacked::<T, W, W>(src),
            Scan::Words => {
                let size!(WP) = comptime!(src.store.packing.physical(src.store.vector_size));
                self.scan_unpacked::<T, WP, W>(src)
            }
        }
    }

    /// Cooperative cyclic flat scan across the units this fill's level names
    /// ([`Level::copied_by`]); a
    /// packed source unpacks.
    fn scan_unpacked<T: Numeric, WP: Size, W: Size>(&mut self, src: &Memory) {
        let fill = comptime!(self.access.copy.clone());
        let s = src.flat_unpacked::<T, WP, W>();
        let mut d = self.flat_mut::<T, W>();
        let total = d.shape();
        // The lines dealt to the units that fill: the deal is disjoint, so a destination that adds
        // (`Accumulate`, `Relay`) takes each value once.
        for line in Walk::copy(&fill, total) {
            let i = line.coord(Level::COPIED);
            // `src` zeroes reads past its bound; the staged buffer writes the full padded cell.
            d.write(i, s.read(i));
        }
    }

    /// A copy of `src` into this window with a cast, `S` down (or up) to `T`, one line for one
    /// line: what a sum held in a wider buffer becomes when it lands in its output. The two are
    /// the same box, plain, and served at one width; the same cyclic scan across the units that
    /// share the window as [`fill_scanned`](Memory::fill_scanned), each casting the lines it moves.
    pub(crate) fn fill_cast_from<T: Numeric, S: Numeric>(
        &mut self,
        src: &Memory,
        #[comptime] space: &Space,
    ) {
        comptime!(assert!(
            src.store.packing.as_stored()
                && self.store.packing.as_stored()
                && src.store.vector_size == self.store.vector_size
                && src.projection.is_direct()
                && self.projection.is_direct(),
            "Memory::fill_cast_from: a cast copy moves plain lines between two boxes served at \
             one width"
        ));
        // The innermost extent is whole lines, as the uncast scan asks (`Scan::new`): nothing on
        // this path would notice the last line the source cannot fill.
        comptime!(fill_extent(
            space,
            src.store.vector_size,
            self.store.vector_size,
            src.access.overhang.masks()
        ));
        let fill = comptime!(self.access.copy.clone());
        let size!(W) = comptime!(self.store.vector_size);
        let s = src.flat_unpacked::<S, W, W>();
        let mut d = self.flat_mut::<T, W>();
        let total = d.shape();
        // Dealt to the units that share this window, as the uncast scan is: a window of one
        // plane's is moved by that plane alone, and a destination that adds takes each line once.
        for line in Walk::copy(&fill, total) {
            let i = line.coord(Level::COPIED);
            d.write(i, Vector::<T, W>::cast_from(s.read(i)));
        }
    }
}
