use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    fmt::Display,
    hash::{BuildHasher, Hash, Hasher},
};
use cubecl_common::format::{DebugRaw, format_str};
use cubecl_ir::AddressType;
use derive_more::{Eq, PartialEq};

use crate::server::{CubeDim, ExecutionMode};

#[macro_export(local_inner_macros)]
/// Create a new storage ID type.
macro_rules! storage_id_type {
    ($name:ident) => {
        /// Storage ID.
        #[derive(Copy, Clone, Hash, PartialEq, Eq, Debug, PartialOrd, Ord)]
        pub struct $name {
            value: usize,
        }

        impl $name {
            /// Create a new ID.
            pub fn new() -> Self {
                use core::sync::atomic::{AtomicUsize, Ordering};

                static COUNTER: AtomicUsize = AtomicUsize::new(0);

                let value = COUNTER.fetch_add(1, Ordering::Relaxed);
                if value == usize::MAX {
                    core::panic!("Memory ID overflowed");
                }
                Self { value }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Reference to a buffer handle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HandleRef<Id> {
    id: Arc<Id>,
    all: Arc<()>,
}

/// Reference to buffer binding.
#[derive(Clone, Debug)]
pub struct BindingRef<Id> {
    id: Id,
    _all: Arc<()>,
}

impl<Id> BindingRef<Id>
where
    Id: Clone + core::fmt::Debug,
{
    /// The id associated to the buffer.
    pub(crate) fn id(&self) -> &Id {
        &self.id
    }
}

impl<Id> HandleRef<Id>
where
    Id: Clone + core::fmt::Debug,
{
    /// Create a new handle.
    pub(crate) fn new(id: Id) -> Self {
        Self {
            id: Arc::new(id),
            all: Arc::new(()),
        }
    }

    /// The id associated to the handle.
    pub(crate) fn id(&self) -> &Id {
        &self.id
    }

    /// Get the binding.
    pub(crate) fn binding(self) -> BindingRef<Id> {
        BindingRef {
            id: self.id.as_ref().clone(),
            _all: self.all,
        }
    }

    /// If the handle can be mut.
    pub(crate) fn can_mut(&self) -> bool {
        // 1 memory management reference with 1 tensor reference.
        Arc::strong_count(&self.id) <= 2
    }

    /// If the resource is free.
    pub(crate) fn is_free(&self) -> bool {
        Arc::strong_count(&self.all) <= 1
    }
}

#[macro_export(local_inner_macros)]
/// Create new memory ID types.
macro_rules! memory_id_type {
    ($id:ident, $handle:ident) => {
        /// Memory Handle.
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $handle {
            value: $crate::id::HandleRef<$id>,
        }

        /// Memory ID.
        #[derive(Clone, Copy, Hash, PartialEq, Eq, Debug)]
        pub struct $id {
            pub(crate) value: usize,
        }

        impl $handle {
            /// Create a new ID.
            pub(crate) fn new() -> Self {
                let value = Self::gen_id();
                Self {
                    value: $crate::id::HandleRef::new($id { value }),
                }
            }

            fn gen_id() -> usize {
                static COUNTER: core::sync::atomic::AtomicUsize =
                    core::sync::atomic::AtomicUsize::new(0);

                let value = COUNTER.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                if value == usize::MAX {
                    core::panic!("Memory ID overflowed");
                }

                value
            }
        }

        impl core::ops::Deref for $handle {
            type Target = $crate::id::HandleRef<$id>;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl Default for $handle {
            fn default() -> Self {
                Self::new()
            }
        }
    };

    ($id:ident, $handle:ident, $binding:ident) => {
        memory_id_type!($id, $handle);

        /// Binding of a memory handle.
        #[derive(Clone, Debug)]
        pub struct $binding {
            value: $crate::id::BindingRef<$id>,
        }

        impl $handle {
            pub(crate) fn binding(self) -> $binding {
                $binding {
                    value: self.value.binding(),
                }
            }
        }

        impl core::ops::Deref for $binding {
            type Target = $crate::id::BindingRef<$id>;

            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }
    };
}

/// Kernel unique identifier.
#[derive(Clone, PartialEq, Eq)]
pub struct KernelId {
    #[eq(skip)]
    type_name: &'static str,
    pub(crate) type_id: core::any::TypeId,
    pub(crate) address_type: AddressType,
    pub(crate) cube_dim: Option<CubeDim>,
    pub(crate) mode: ExecutionMode,
    pub(crate) info: Option<Info>,
}

impl Hash for KernelId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        self.address_type.hash(state);
        self.cube_dim.hash(state);
        self.mode.hash(state);
        self.info.hash(state);
    }
}

impl core::fmt::Debug for KernelId {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut debug_str = f.debug_struct("KernelId");
        debug_str
            .field("type", &DebugRaw(self.type_name))
            .field("address_type", &self.address_type);
        match &self.cube_dim {
            Some(cube_dim) => debug_str.field("cube_dim", cube_dim),
            None => debug_str.field("cube_dim", &self.cube_dim),
        };
        debug_str.field("mode", &self.mode);
        match &self.info {
            Some(info) => debug_str.field("info", info),
            None => debug_str.field("info", &self.info),
        };
        debug_str.finish()
    }
}

impl Display for KernelId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.info {
            Some(info) => f.write_str(
                format_str(
                    format!("{info:?}").as_str(),
                    &[('(', ')'), ('[', ']'), ('{', '}')],
                    true,
                )
                .as_str(),
            ),
            None => f.write_str("No info"),
        }
    }
}

impl KernelId {
    /// Create a new [kernel id](KernelId) for a type.
    pub fn new<T: 'static>() -> Self {
        Self {
            type_id: core::any::TypeId::of::<T>(),
            type_name: core::any::type_name::<T>(),
            info: None,
            cube_dim: None,
            mode: ExecutionMode::Checked,
            address_type: Default::default(),
        }
    }

    /// Render the key in a standard format that can be used between runs.
    ///
    /// Can be used as a persistent kernel cache key.
    pub fn stable_format(&self) -> String {
        format!(
            "{}-{}-{:?}-{:?}-{:?}",
            self.type_name, self.address_type, self.cube_dim, self.mode, self.info
        )
    }

    /// Add information to the [kernel id](KernelId).
    ///
    /// The information is used to differentiate kernels of the same kind but with different
    /// configurations, which affect the generated code.
    pub fn info<I: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(
        mut self,
        info: I,
    ) -> Self {
        self.info = Some(Info::new(info));
        self
    }

    /// Set the [execution mode](ExecutionMode).
    pub fn mode(&mut self, mode: ExecutionMode) {
        self.mode = mode;
    }

    /// Set the [cube dim](CubeDim).
    pub fn cube_dim(mut self, cube_dim: CubeDim) -> Self {
        self.cube_dim = Some(cube_dim);
        self
    }

    /// Set the [address_type](AddressType).
    pub fn address_type(mut self, addr_ty: AddressType) -> Self {
        self.address_type = addr_ty;
        self
    }
}

impl core::fmt::Debug for Info {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.value.fmt(f)
    }
}

impl Info {
    fn new<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(id: T) -> Self {
        Self {
            value: Arc::new(id),
        }
    }
}

/// This trait allows various types to be used as keys within a single data structure.
///
/// The downside is that the hashing method is hardcoded and cannot be configured using the
/// [core::hash::Hash] function. The provided [Hasher] will be modified, but only based on the
/// result of the hash from the [DefaultHasher].
trait DynKey: core::fmt::Debug + Send + Sync {
    fn dyn_type_id(&self) -> TypeId;
    fn dyn_eq(&self, other: &dyn DynKey) -> bool;
    fn dyn_hash(&self, state: &mut dyn Hasher);
    fn as_any(&self) -> &dyn Any;
}

impl PartialEq for Info {
    fn eq(&self, other: &Self) -> bool {
        self.value.dyn_eq(other.value.as_ref())
    }
}

/// Extra information
#[derive(Clone)]
pub(crate) struct Info {
    value: Arc<dyn DynKey>,
}
impl Eq for Info {}

impl Hash for Info {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.dyn_type_id().hash(state);
        self.value.dyn_hash(state)
    }
}

impl<T: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync> DynKey for T {
    fn dyn_eq(&self, other: &dyn DynKey) -> bool {
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
        // HashBrown uses foldhash but the default hasher still creates some random state. We need this hash here
        // to be exactly reproducible.
        let hash = foldhash::fast::FixedState::with_seed(0).hash_one(self);
        state.write_u64(hash);
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
        let value_1 = KernelId::new::<()>().info("1");
        let value_2 = KernelId::new::<()>().info("2");

        let mut set = HashSet::new();

        set.insert(value_1.clone());

        assert!(set.contains(&value_1));
        assert!(!set.contains(&value_2));
    }
}
