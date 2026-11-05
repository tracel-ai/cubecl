//! [`Stagger`]: how far apart a buffer's coarsest pieces start.

/// The rule a [`StoragePartitioning`](super::StoragePartitioning) lays its coarsest pieces down
/// by: each starts a whole and odd number of memory transactions after the one before it, the run
/// of what is finer padded up to that.
///
/// A memory system that picks its channel from low address bits sends pieces a power of two bytes
/// apart to one channel, and a read walking every piece at the same offset then waits on that one
/// channel; an odd step reaches every slot of such an interleave. A weight in storage tiles is
/// such a walk: its strips of tiles lie a power of two bytes apart whenever its depth is one, and
/// a product reads them all at once.
///
/// Stated in bits, so that two statements of one buffer counted in different elements, a packed
/// matrix's values and the words holding them, pad it to the same bytes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Stagger {
    /// Bits one memory transaction moves.
    pub transaction_bits: usize,
    /// Bits one element of the statement holds.
    pub element_bits: usize,
}

impl Stagger {
    /// The stride, in elements, between pieces each `run` elements long: the fewest whole
    /// transactions that hold the run, an odd number of them. A run already an odd number of
    /// transactions keeps its length.
    ///
    /// # Panics
    ///
    /// When an element does not divide a transaction.
    pub fn pitch(&self, run: usize) -> usize {
        assert!(
            self.transaction_bits.is_multiple_of(self.element_bits),
            "Stagger::pitch: a {}-bit element does not divide a {}-bit transaction",
            self.element_bits,
            self.transaction_bits
        );
        let transactions = (run * self.element_bits)
            .div_ceil(self.transaction_bits)
            .max(1);
        (transactions | 1) * self.transaction_bits / self.element_bits
    }

    /// The same stagger over elements of `element_bits`.
    pub fn over(self, element_bits: usize) -> Self {
        Self {
            element_bits,
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 128-byte transactions over `e2m1` values.
    const VALUES: Stagger = Stagger {
        transaction_bits: 1024,
        element_bits: 4,
    };

    #[test]
    fn a_power_of_two_run_takes_one_more_transaction() {
        // 2048 bytes is sixteen transactions; seventeen hold it.
        assert_eq!(VALUES.pitch(4096), 17 * 256);
    }

    #[test]
    fn an_odd_run_keeps_its_length() {
        assert_eq!(VALUES.pitch(3 * 256), 3 * 256);
    }

    #[test]
    fn a_short_run_takes_a_whole_transaction() {
        assert_eq!(VALUES.pitch(10), 256);
    }

    /// The values and the `u32` words packing them, eight a word, pad to the same bytes.
    #[test]
    fn every_statement_of_one_buffer_pads_it_alike() {
        let words = VALUES.over(32);
        for run in [4096, 12288 * 16, 2560 * 16, 768] {
            assert_eq!(
                VALUES.pitch(run) * 4,
                words.pitch(run / 8) * 32,
                "run {run}"
            );
        }
    }
}
