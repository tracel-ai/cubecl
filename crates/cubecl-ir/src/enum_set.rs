//! A compact set of enum variants, stored as a bitmask.
//!
//! [`EnumSet`] is a drop-in stand-in for the `enumset` crate's type of the same name, kept in the
//! tree so the derive can share `cubecl-macros-internal`'s `darling`/`syn` rather than pulling a
//! second, older copy of both through `enumset_derive`.

use core::fmt::{self, Debug};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{
    BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Sub, SubAssign,
};

/// An enum whose variants can be collected into an [`EnumSet`].
///
/// Derive it with `#[derive(EnumSetType)]`, alongside `Clone`, `Copy`, `PartialEq` and `Eq`.
///
/// The set is backed by a `u64`, so an enum may have at most 64 variants, and it must be a plain
/// C-like enum with no fields.
pub trait EnumSetType: Copy + Eq + 'static {
    /// Number of variants in the enum.
    const VARIANTS: u32;

    /// Index of the bit representing this variant.
    fn to_bit(self) -> u32;

    /// The variant a bit index stands for.
    ///
    /// # Panics
    /// If `bit` is not below [`Self::VARIANTS`].
    fn from_bit(bit: u32) -> Self;
}

/// A set of enum variants, stored as a bitmask.
pub struct EnumSet<T: EnumSetType> {
    bits: u64,
    _ty: PhantomData<T>,
}

impl<T: EnumSetType> EnumSet<T> {
    /// Mask with every valid bit for `T` set.
    const MASK: u64 = if T::VARIANTS >= 64 {
        u64::MAX
    } else {
        (1u64 << T::VARIANTS) - 1
    };

    /// The empty set.
    pub const fn empty() -> Self {
        Self {
            bits: 0,
            _ty: PhantomData,
        }
    }

    /// The set of every variant of `T`.
    pub const fn all() -> Self {
        Self {
            bits: Self::MASK,
            _ty: PhantomData,
        }
    }

    /// A set holding only `value`.
    pub fn only(value: T) -> Self {
        Self {
            bits: 1 << value.to_bit(),
            _ty: PhantomData,
        }
    }

    /// The raw bitmask.
    pub const fn as_u64(self) -> u64 {
        self.bits
    }

    /// Build a set from a raw bitmask, discarding bits that name no variant.
    pub const fn from_u64_truncated(bits: u64) -> Self {
        Self {
            bits: bits & Self::MASK,
            _ty: PhantomData,
        }
    }

    /// Number of variants in the set.
    pub const fn len(self) -> usize {
        self.bits.count_ones() as usize
    }

    /// Whether the set holds no variants.
    pub const fn is_empty(self) -> bool {
        self.bits == 0
    }

    /// Whether `value` is in the set.
    pub fn contains(self, value: T) -> bool {
        self.bits & (1 << value.to_bit()) != 0
    }

    /// Add `value`, returning whether it was newly added.
    pub fn insert(&mut self, value: T) -> bool {
        let added = !self.contains(value);
        self.bits |= 1 << value.to_bit();
        added
    }

    /// Remove `value`, returning whether it had been present.
    pub fn remove(&mut self, value: T) -> bool {
        let present = self.contains(value);
        self.bits &= !(1 << value.to_bit());
        present
    }

    /// Remove every variant.
    pub fn clear(&mut self) {
        self.bits = 0;
    }

    /// Variants in either set.
    pub const fn union(self, other: Self) -> Self {
        Self {
            bits: self.bits | other.bits,
            _ty: PhantomData,
        }
    }

    /// Variants in both sets.
    pub const fn intersection(self, other: Self) -> Self {
        Self {
            bits: self.bits & other.bits,
            _ty: PhantomData,
        }
    }

    /// Variants in `self` but not `other`.
    pub const fn difference(self, other: Self) -> Self {
        Self {
            bits: self.bits & !other.bits,
            _ty: PhantomData,
        }
    }

    /// Variants in exactly one of the two sets.
    pub const fn symmetrical_difference(self, other: Self) -> Self {
        Self {
            bits: self.bits ^ other.bits,
            _ty: PhantomData,
        }
    }

    /// Every variant not in the set.
    pub const fn complement(self) -> Self {
        Self {
            bits: !self.bits & Self::MASK,
            _ty: PhantomData,
        }
    }

    /// Whether every variant of `self` is in `other`.
    pub const fn is_subset(self, other: Self) -> bool {
        self.bits & other.bits == self.bits
    }

    /// Whether every variant of `other` is in `self`.
    pub const fn is_superset(self, other: Self) -> bool {
        other.is_subset(self)
    }

    /// Whether the two sets share no variants.
    pub const fn is_disjoint(self, other: Self) -> bool {
        self.bits & other.bits == 0
    }

    /// Iterate the variants in the set, in declaration order.
    pub fn iter(self) -> EnumSetIter<T> {
        EnumSetIter {
            bits: self.bits,
            _ty: PhantomData,
        }
    }
}

impl<T: EnumSetType> Clone for EnumSet<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: EnumSetType> Copy for EnumSet<T> {}

impl<T: EnumSetType> Default for EnumSet<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: EnumSetType> PartialEq for EnumSet<T> {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl<T: EnumSetType> Eq for EnumSet<T> {}

impl<T: EnumSetType> PartialOrd for EnumSet<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: EnumSetType> Ord for EnumSet<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.bits.cmp(&other.bits)
    }
}

impl<T: EnumSetType> Hash for EnumSet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}

impl<T: EnumSetType + Debug> Debug for EnumSet<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EnumSet(")?;
        for (i, value) in self.iter().enumerate() {
            if i > 0 {
                write!(f, " | ")?;
            }
            write!(f, "{value:?}")?;
        }
        write!(f, ")")
    }
}

impl<T: EnumSetType> From<T> for EnumSet<T> {
    fn from(value: T) -> Self {
        Self::only(value)
    }
}

impl<T: EnumSetType> FromIterator<T> for EnumSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = Self::empty();
        set.extend(iter);
        set
    }
}

impl<T: EnumSetType> Extend<T> for EnumSet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.insert(value);
        }
    }
}

impl<T: EnumSetType> IntoIterator for EnumSet<T> {
    type Item = T;
    type IntoIter = EnumSetIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over the variants of an [`EnumSet`], in declaration order.
pub struct EnumSetIter<T: EnumSetType> {
    bits: u64,
    _ty: PhantomData<T>,
}

impl<T: EnumSetType> Iterator for EnumSetIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.bits == 0 {
            return None;
        }
        let bit = self.bits.trailing_zeros();
        self.bits &= self.bits - 1;
        Some(T::from_bit(bit))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bits.count_ones() as usize;
        (len, Some(len))
    }
}

impl<T: EnumSetType> ExactSizeIterator for EnumSetIter<T> {}

/// Generates the set/set and set/variant halves of a binary operator.
macro_rules! impl_op {
    ($op: ident, $method: ident, $assign_op: ident, $assign_method: ident, $call: ident) => {
        impl<T: EnumSetType> $op<EnumSet<T>> for EnumSet<T> {
            type Output = EnumSet<T>;

            fn $method(self, other: EnumSet<T>) -> EnumSet<T> {
                self.$call(other)
            }
        }

        impl<T: EnumSetType> $op<T> for EnumSet<T> {
            type Output = EnumSet<T>;

            fn $method(self, other: T) -> EnumSet<T> {
                self.$call(EnumSet::only(other))
            }
        }

        impl<T: EnumSetType> $assign_op<EnumSet<T>> for EnumSet<T> {
            fn $assign_method(&mut self, other: EnumSet<T>) {
                *self = self.$call(other);
            }
        }

        impl<T: EnumSetType> $assign_op<T> for EnumSet<T> {
            fn $assign_method(&mut self, other: T) {
                *self = self.$call(EnumSet::only(other));
            }
        }
    };
}

impl_op!(BitOr, bitor, BitOrAssign, bitor_assign, union);
impl_op!(BitAnd, bitand, BitAndAssign, bitand_assign, intersection);
impl_op!(
    BitXor,
    bitxor,
    BitXorAssign,
    bitxor_assign,
    symmetrical_difference
);
impl_op!(Sub, sub, SubAssign, sub_assign, difference);

impl<T: EnumSetType> Not for EnumSet<T> {
    type Output = EnumSet<T>;

    fn not(self) -> EnumSet<T> {
        self.complement()
    }
}

#[cfg(feature = "serde")]
impl<T: EnumSetType> serde::Serialize for EnumSet<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.bits)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: EnumSetType> serde::Deserialize<'de> for EnumSet<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error;

        let bits = u64::deserialize(deserializer)?;
        if bits & !Self::MASK != 0 {
            return Err(D::Error::custom(
                "bitmask holds bits that name no variant of the enum",
            ));
        }
        Ok(Self {
            bits,
            _ty: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_macros_internal::EnumSetType;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, EnumSetType)]
    enum Usage {
        Load,
        Store,
        Arithmetic,
    }

    #[test]
    fn all_holds_every_variant() {
        let all = EnumSet::<Usage>::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(Usage::Load));
        assert!(all.contains(Usage::Store));
        assert!(all.contains(Usage::Arithmetic));
    }

    #[test]
    fn empty_is_the_default() {
        assert_eq!(EnumSet::<Usage>::default(), EnumSet::empty());
        assert!(EnumSet::<Usage>::empty().is_empty());
        assert_eq!(EnumSet::<Usage>::empty().len(), 0);
    }

    #[test]
    fn or_on_variants_builds_a_set() {
        let set = Usage::Load | Usage::Store;
        assert_eq!(set.len(), 2);
        assert!(set.contains(Usage::Load));
        assert!(!set.contains(Usage::Arithmetic));
    }

    #[test]
    fn insert_and_remove_report_whether_the_set_changed() {
        let mut set = EnumSet::empty();
        assert!(set.insert(Usage::Load));
        assert!(!set.insert(Usage::Load));
        assert!(set.remove(Usage::Load));
        assert!(!set.remove(Usage::Load));
    }

    #[test]
    fn iter_yields_declaration_order() {
        let set = Usage::Arithmetic | Usage::Load;
        let seen: alloc::vec::Vec<_> = set.iter().collect();
        assert_eq!(seen, alloc::vec![Usage::Load, Usage::Arithmetic]);
    }

    #[test]
    fn complement_stays_inside_the_variant_mask() {
        let set = EnumSet::only(Usage::Load);
        assert_eq!(!set, Usage::Store | Usage::Arithmetic);
        assert_eq!((!EnumSet::<Usage>::empty()), EnumSet::all());
    }

    #[test]
    fn set_algebra() {
        let a = Usage::Load | Usage::Store;
        let b = Usage::Store | Usage::Arithmetic;
        assert_eq!(a.intersection(b), EnumSet::only(Usage::Store));
        assert_eq!(a.difference(b), EnumSet::only(Usage::Load));
        assert_eq!(a.union(b), EnumSet::all());
        assert_eq!(a.symmetrical_difference(b), Usage::Load | Usage::Arithmetic);
        assert!(EnumSet::only(Usage::Load).is_subset(a));
        assert!(a.is_superset(EnumSet::only(Usage::Load)));
        assert!(EnumSet::only(Usage::Load).is_disjoint(EnumSet::only(Usage::Store)));
    }
}
