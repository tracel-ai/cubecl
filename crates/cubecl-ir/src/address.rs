use crate::{IntKind, Scope, StorageType, UIntKind};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default, PartialOrd, Ord)]
pub enum AddressType {
    #[default]
    U32,
    U64,
}

impl AddressType {
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
