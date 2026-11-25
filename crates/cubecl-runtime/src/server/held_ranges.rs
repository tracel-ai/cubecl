use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::ops::Range;

/// The byte ranges of a lazy read that its bytes and their views hold, one count per holder.
#[derive(Debug, Default)]
pub(crate) struct HeldRanges {
    /// Ordered usage: how many holders each range has, by start then end, which merging reads in
    /// byte order.
    holders: BTreeMap<(usize, usize), usize>,
}

impl HeldRanges {
    /// One more holder of `range`.
    pub(crate) fn hold(&mut self, range: Range<usize>) {
        *self.holders.entry((range.start, range.end)).or_default() += 1;
    }

    /// One holder of `range` fewer.
    pub(crate) fn let_go(&mut self, range: &Range<usize>) {
        let key = (range.start, range.end);
        let Some(holders) = self.holders.get_mut(&key) else {
            return;
        };
        *holders -= 1;
        if *holders == 0 {
            self.holders.remove(&key);
        }
    }

    /// The bytes held, in byte order, with overlapping and touching ranges merged and empty ones
    /// left out.
    pub(crate) fn merged(&self) -> Vec<Range<usize>> {
        let mut merged: Vec<Range<usize>> = Vec::new();
        let held = self.holders.keys().filter(|(start, end)| start < end);
        for &(start, end) in held {
            match merged.last_mut() {
                Some(last) if start <= last.end => last.end = last.end.max(end),
                _ => merged.push(start..end),
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn merged_joins_touching_ranges_and_drops_empty_ones() {
        let mut held = HeldRanges::default();
        held.hold(4..8);
        held.hold(0..4);
        held.hold(6..6);
        held.hold(10..12);

        assert_eq!(held.merged(), vec![0..8, 10..12]);
    }

    #[test]
    fn a_range_held_twice_stays_held_after_one_let_go() {
        let mut held = HeldRanges::default();
        held.hold(0..4);
        held.hold(0..4);

        held.let_go(&(0..4));

        assert_eq!(held.merged(), vec![0..4]);
    }
}
