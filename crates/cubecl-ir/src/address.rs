use crate::{IntKind, Scope, StorageType, UIntKind};

/// The type used for addressing storage types in a kernel.
/// This is the type `usize` maps to when used in a kernel, with `isize` being mapped to the signed
/// equivalent.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default, PartialOrd, Ord)]
pub enum AddressType {
    // Discriminants are explicit to ensure correct ordering
    #[default]
    U32 = 0,
    U64 = 1,
}

impl core::fmt::Display for AddressType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AddressType::U32 => f.write_str("u32"),
            AddressType::U64 => f.write_str("u64"),
        }
    }
}

impl AddressType {
    /// Pick an address type based on the number of elements in a buffer.
    pub fn from_len(num_elems: usize) -> Self {
        if num_elems > u32::MAX as usize {
            AddressType::U64
        } else {
            AddressType::U32
        }
    }

    /// Pick an address type based on the number of elements in a buffer, for a kernel that requires
    /// signed indices.
    pub fn from_len_signed(num_elems: usize) -> Self {
        if num_elems > i32::MAX as usize {
            AddressType::U64
        } else {
            AddressType::U32
        }
    }

    pub fn register(&self, scope: &mut Scope) {
        scope.register_type::<usize>(self.unsigned_type());
        scope.register_type::<isize>(self.signed_type());
    }

    pub fn unsigned_type(&self) -> StorageType {
        match self {
            AddressType::U32 => UIntKind::U32.into(),
            AddressType::U64 => UIntKind::U64.into(),
        }
    }

    pub fn signed_type(&self) -> StorageType {
        match self {
            AddressType::U32 => IntKind::I32.into(),
            AddressType::U64 => IntKind::I64.into(),
        }
    }
}
