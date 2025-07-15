use alloc::format;
use alloc::string::String;
use alloc::string::ToString;
use alloc::sync::Arc;
use core::{
    any::{Any, TypeId},
    fmt::Display,
    hash::{BuildHasher, Hash, Hasher},
};
use cubecl_common::ExecutionMode;

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
#[derive(Clone, Debug)]
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
        #[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
pub struct KernelId {
    pub(crate) type_id: core::any::TypeId,
    pub(crate) info: Option<Info>,
    pub(crate) mode: Option<ExecutionMode>,
    type_name: &'static str,
}

impl Hash for KernelId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        self.info.hash(state);
        self.mode.hash(state);
    }
}

impl PartialEq for KernelId {
    fn eq(&self, other: &Self) -> bool {
        self.type_id == other.type_id && self.mode == other.mode && self.info == other.info
    }
}

impl Eq for KernelId {}

/// Format strings for use in identifiers and types.
pub fn format_str(kernel_id: &str, markers: &[(char, char)], include_space: bool) -> String {
    let kernel_id = kernel_id.to_string();
    let mut result = String::new();
    let mut depth = 0;
    let indentation = 4;

    let mut prev = ' ';

    for c in kernel_id.chars() {
        if c == ' ' {
            continue;
        }

        let mut found_marker = false;

        for (start, end) in markers {
            let (start, end) = (*start, *end);

            if c == start {
                depth += 1;
                if prev != ' ' && include_space {
                    result.push(' ');
                }
                result.push(start);
                result.push('\n');
                result.push_str(&" ".repeat(indentation * depth));
                found_marker = true;
            } else if c == end {
                depth -= 1;
                if prev != start {
                    if prev == ' ' {
                        result.pop();
                    }
                    result.push_str(",\n");
                    result.push_str(&" ".repeat(indentation * depth));
                    result.push(end);
                } else {
                    for _ in 0..(&" ".repeat(indentation * depth).len()) + 1 + indentation {
                        result.pop();
                    }
                    result.push(end);
                }
                found_marker = true;
            }
        }

        if found_marker {
            prev = c;
            continue;
        }

        if c == ',' && depth > 0 {
            if prev == ' ' {
                result.pop();
            }

            result.push_str(",\n");
            result.push_str(&" ".repeat(indentation * depth));
            continue;
        }

        if c == ':' && include_space {
            result.push(c);
            result.push(' ');
            prev = ' ';
        } else {
            result.push(c);
            prev = c;
        }
    }

    result
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
            mode: None,
        }
    }

    /// Render the key in a standard format that can be used between runs.
    ///
    /// Can be used as a persistent kernel cache key.
    pub fn stable_format(&self) -> String {
        format!("{}-{:?}-{:?}", self.type_name, self.info, self.mode)
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
        self.mode = Some(mode);
    }
}

impl core::fmt::Debug for Info {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.value))
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

/// The device id.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, new)]
pub struct DeviceId {
    /// The type id identifies the type of the device.
    pub type_id: u16,
    /// The index id identifies the device number.
    pub index_id: u32,
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
