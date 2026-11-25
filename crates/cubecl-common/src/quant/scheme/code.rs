/// Fields packed one after the other into a `u64`, from its low bits.
#[derive(Default)]
pub(super) struct PackedFields {
    pub bits: u64,
    pub used_bits: u32,
}

impl PackedFields {
    /// With `field` in the next `width` bits; `None` where the field or the code is too narrow to
    /// hold it.
    pub fn with(self, field: u64, width: u32) -> Option<Self> {
        let used_bits = self.used_bits + width;
        (field >> width == 0 && used_bits <= u64::BITS).then(|| Self {
            bits: self.bits | (field << self.used_bits),
            used_bits,
        })
    }

    /// The next field of `width` bits, and the fields after it.
    pub fn split_bits(self, width: u32) -> Option<(u64, Self)> {
        let used_bits = self.used_bits + width;
        (used_bits <= u64::BITS).then(|| {
            let field = (self.bits >> self.used_bits) & ((1 << width) - 1);
            (field, Self { used_bits, ..self })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_field_goes_in_the_bits_after_the_used_ones() {
        let packed = PackedFields {
            bits: 5,
            used_bits: 3,
        }
        .with(1, 2)
        .unwrap();
        assert_eq!((packed.bits, packed.used_bits), (0b01_101, 5));
    }

    #[test]
    fn a_field_is_read_from_the_bits_after_the_used_ones() {
        let code = PackedFields {
            bits: 0b01_101,
            used_bits: 3,
        };
        let (field, rest) = code.split_bits(2).unwrap();
        assert_eq!((field, rest.used_bits), (1, 5));
    }

    #[test]
    fn a_field_past_its_width_is_refused() {
        assert!(PackedFields::default().with(8, 3).is_none());
    }

    #[test]
    fn a_field_past_the_code_is_refused() {
        let full = PackedFields {
            bits: 0,
            used_bits: 62,
        };
        assert!(full.with(1, 3).is_none());
    }

    #[test]
    fn a_read_past_the_code_is_refused() {
        let full = PackedFields {
            bits: 0,
            used_bits: 62,
        };
        assert!(full.split_bits(3).is_none());
    }
}
