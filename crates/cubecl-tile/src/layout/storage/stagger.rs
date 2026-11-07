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
    transaction_bits: usize,
    /// Bits one element of the statement holds, a divisor of the transaction's.
    element_bits: usize,
}

impl Stagger {
    /// The stagger by transactions of `transaction_bits` over elements of `element_bits`; `None`
    /// where an element does not divide a transaction, which no whole number of elements pads.
    pub fn new(transaction_bits: usize, element_bits: usize) -> Option<Self> {
        let divides = element_bits > 0 && transaction_bits.is_multiple_of(element_bits);
        divides.then_some(Self {
            transaction_bits,
            element_bits,
        })
    }

    /// Bits one memory transaction moves.
    pub fn transaction_bits(&self) -> usize {
        self.transaction_bits
    }

    /// The stride, in elements, between pieces each `run` elements long: the fewest whole
    /// transactions that hold the run, an odd number of them. A run already an odd number of
    /// transactions keeps its length.
    pub fn pitch(&self, run: usize) -> usize {
        let transactions = (run * self.element_bits)
            .div_ceil(self.transaction_bits)
            .max(1);
        (transactions | 1) * self.transaction_bits / self.element_bits
    }

    /// The same stagger over elements of `element_bits`; `None` where one does not divide a
    /// transaction.
    pub fn over(self, element_bits: usize) -> Option<Self> {
        Self::new(self.transaction_bits, element_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 128-byte transactions over `e2m1` values.
    fn values() -> Stagger {
        Stagger::new(1024, 4).expect("a 4-bit value divides a 1024-bit transaction")
    }

    /// An element that does not divide a transaction is no stagger.
    #[test]
    fn an_element_that_does_not_divide_a_transaction_is_refused() {
        assert_eq!(Stagger::new(1024, 3), None);
        assert_eq!(Stagger::new(1024, 0), None);
        assert_eq!(values().over(3), None);
    }

    #[test]
    fn a_power_of_two_run_takes_one_more_transaction() {
        // 2048 bytes is sixteen transactions; seventeen hold it.
        assert_eq!(values().pitch(4096), 17 * 256);
    }

    #[test]
    fn an_odd_run_keeps_its_length() {
        assert_eq!(values().pitch(3 * 256), 3 * 256);
    }

    #[test]
    fn a_short_run_takes_a_whole_transaction() {
        assert_eq!(values().pitch(10), 256);
    }

    /// The values and the `u32` words packing them, eight a word, pad to the same bytes.
    #[test]
    fn every_statement_of_one_buffer_pads_it_alike() {
        let words = values().over(32).expect("a word divides a transaction");
        for run in [4096, 12288 * 16, 2560 * 16, 768] {
            assert_eq!(
                values().pitch(run) * 4,
                words.pitch(run / 8) * 32,
                "run {run}"
            );
        }
    }
}
