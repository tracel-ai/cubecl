use std::any::{Any, TypeId};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use cubecl_common::ExecutionMode;
use cubecl_runtime::client::ComputeClient;

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

/// Extra information
#[derive(Clone)]
pub(crate) struct Info {
    value: Arc<dyn DynKey>,
}

impl core::fmt::Debug for Info {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        let mut default_hasher = DefaultHasher::new();
        self.hash(&mut default_hasher);
        state.write_u64(default_hasher.finish());
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

/// ID used to identify a Just-in-Time environment.
#[derive(Hash, PartialEq, Eq, Debug, Clone)]
pub struct CubeTuneId {
    device: DeviceId,
    name: &'static str,
}

impl CubeTuneId {
    /// Create a new ID.
    pub fn new<R: crate::Runtime>(
        client: &ComputeClient<R::Server, R::Channel>,
        device: &R::Device,
    ) -> Self {
        Self {
            device: R::device_id(device),
            name: R::name(client),
        }
    }
}

impl core::fmt::Display for CubeTuneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "device-{}-{}-{}",
            self.device.type_id, self.device.index_id, self.name
        ))
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
