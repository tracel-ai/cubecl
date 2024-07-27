use std::any::{Any, TypeId};
use std::fmt::Display;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

/// Kernel unique identifier.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct KernelId {
    type_id: core::any::TypeId,
    info: Info,
}

impl Display for KernelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.info))
    }
}

impl KernelId {
    /// Create a new [kernel id](KernelId) for a type.
    pub fn new<T: 'static, I: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(
        info: I,
    ) -> Self {
        Self {
            type_id: core::any::TypeId::of::<T>(),
            info: Info::new(info),
        }
    }
}

/// Extra information
#[derive(Clone, Debug)]
struct Info {
    id: Arc<dyn DynId>,
}

impl Info {
    fn new<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(id: T) -> Self {
        Self { id: Arc::new(id) }
    }
}

trait DynId: core::fmt::Debug + Send + Sync {
    fn dyn_type_id(&self) -> TypeId;
    fn dyn_eq(&self, other: &dyn DynId) -> bool;
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn as_any(&self) -> &dyn Any;
}

impl PartialEq for Info {
    fn eq(&self, other: &Self) -> bool {
        self.id.dyn_eq(other.id.as_ref())
    }
}

impl Eq for Info {}

impl Hash for Info {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.dyn_type_id().hash(state);
        self.id.dyn_hash(state)
    }
}

impl<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync> DynId for T {
    fn dyn_eq(&self, other: &dyn DynId) -> bool {
        if let Some(other) = other.as_any().downcast_ref::<T>() {
            self == other
        } else {
            false
        }
    }

    fn dyn_type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        let mut default_hasher = DefaultHasher::new();
        self.hash(&mut default_hasher);
        state.write_u64(default_hasher.finish());
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    pub fn kernel_id_hash() {
        let value_1 = KernelId::new::<(), _>("1");
        let value_2 = KernelId::new::<(), _>("2");

        let mut set = HashSet::new();

        set.insert(value_1.clone());

        assert!(set.contains(&value_1));
        assert!(!set.contains(&value_2));
    }
}
