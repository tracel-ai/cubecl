use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    hash::{Hash, Hasher},
};
use cubecl_common::hash::{StableHash, StableHasher};

/// The comptime arguments that tell apart kernels of one type, held as a value of any type that
/// can be compared, hashed and printed.
#[derive(Clone)]
pub struct Info {
    value: Arc<dyn Keyable>,
}

impl Info {
    /// Comptime arguments compared, hashed and printed as `value` is.
    pub fn new<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(
        value: T,
    ) -> Self {
        Self {
            value: Arc::new(value),
        }
    }
}

impl core::fmt::Debug for Info {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.value.fmt(f)
    }
}

impl PartialEq for Info {
    fn eq(&self, other: &Self) -> bool {
        self.value.dyn_eq(other.value.as_ref())
    }
}

impl Eq for Info {}

impl Hash for Info {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.dyn_type_id().hash(state);
        self.value.dyn_hash(state)
    }
}

/// What can be held behind a `dyn` and still compared, hashed and printed, so values of
/// different types can be keys of one collection.
///
/// The hash is the value's own [`StableHash`], written into the given [`Hasher`], so it cannot
/// follow that hasher's own algorithm.
trait Keyable: core::fmt::Debug + Send + Sync {
    fn dyn_type_id(&self) -> TypeId;
    fn dyn_eq(&self, other: &dyn Keyable) -> bool;
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn dyn_hash_one(&self) -> StableHash;
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync> Keyable for T {
    fn dyn_eq(&self, other: &dyn Keyable) -> bool {
        other
            .as_any()
            .downcast_ref::<T>()
            .is_some_and(|other| self == other)
    }

    fn dyn_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u128(self.dyn_hash_one());
    }

    fn dyn_hash_one(&self) -> StableHash {
        let mut hasher = StableHasher::new();
        self.hash(&mut hasher);
        hasher.finalize()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_values_are_equal_infos() {
        assert_eq!(Info::new(1u32), Info::new(1u32));
    }

    #[test]
    fn values_of_different_types_are_different_infos() {
        assert_ne!(Info::new(1u32), Info::new(1u64));
    }

    #[test]
    fn equal_values_hash_equal() {
        let state = std::collections::hash_map::RandomState::new();

        assert_eq!(
            core::hash::BuildHasher::hash_one(&state, Info::new(1u32)),
            core::hash::BuildHasher::hash_one(&state, Info::new(1u32))
        );
    }

    #[test]
    fn an_info_prints_its_value() {
        assert_eq!(alloc::format!("{:?}", Info::new("a")), "\"a\"");
    }
}
