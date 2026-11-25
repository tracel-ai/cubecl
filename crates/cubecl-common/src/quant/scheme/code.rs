/// Fields packed one after the other into a `u64`, from its low bits.
#[derive(Default)]
pub(super) struct PackedFields {
    pub bits: u64,
    pub used: u32,
}

impl PackedFields {
    /// With `field` in the next `width` bits; `None` where the field or the code is too narrow to
    /// hold it.
    pub fn with(self, field: u64, width: u32) -> Option<Self> {
        let used = self.used + width;
        (field >> width == 0 && used <= u64::BITS).then(|| Self {
            bits: self.bits | (field << self.used),
            used,
        })
    }

    /// The next field of `width` bits, and the fields after it.
    pub fn split_bits(self, width: u32) -> Option<(u64, Self)> {
        let used = self.used + width;
        (used <= u64::BITS).then(|| {
            let field = (self.bits >> self.used) & ((1 << width) - 1);
            (field, Self { used, ..self })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_read_back_in_order() {
        let packed = PackedFields::default()
            .with(5, 3)
            .unwrap()
            .with(1, 2)
            .unwrap();
        let code = PackedFields {
            bits: packed.bits,
            used: 0,
        };
        let (first, rest) = code.split_bits(3).unwrap();
        assert_eq!((first, rest.split_bits(2).unwrap().0), (5, 1));
    }

    #[test]
    fn a_field_past_its_width_is_refused() {
        assert!(PackedFields::default().with(8, 3).is_none());
    }

    #[test]
    fn a_field_past_the_code_is_refused() {
        let full = PackedFields { bits: 0, used: 62 };
        assert!(full.with(1, 3).is_none());
    }

    #[test]
    fn a_read_past_the_code_is_refused() {
        let full = PackedFields { bits: 0, used: 62 };
        assert!(full.split_bits(3).is_none());
    }
}
