use super::Info;
use crate::config::short_type_name;
use alloc::format;
use alloc::string::String;
use core::{
    fmt::Display,
    hash::{Hash, Hasher},
};
use cubecl_common::{
    format::{DebugRaw, format_str},
    hash::{StableHash, StableHasher},
};
use cubecl_ir::{
    AddressType,
    settings::{DebugInfo, Dim3, ExecutionMode, FloatContraction},
};
use derive_more::{Eq, PartialEq};

/// Everything that tells one compiled kernel from another: its type, its launch settings and its
/// comptime arguments.
#[derive(Clone, PartialEq, Eq)]
pub struct KernelId {
    /// The kernel's type name, for reports and the stable keys. Two ids of one type always
    /// share it, so equality skips it.
    #[eq(skip)]
    pub type_name: &'static str,
    /// The kernel's type.
    pub type_id: core::any::TypeId,
    /// The cube dim the kernel is compiled for.
    pub cube_dim: Dim3,
    /// The type the kernel's addresses are computed in.
    pub address_type: AddressType,
    /// Whether the kernel checks its accesses.
    pub mode: ExecutionMode,
    /// The debug data of this kernel, after the configuration lowered it.
    pub debug_info: DebugInfo,
    /// Whether float multiply-adds may be fused in this kernel.
    pub float_contraction: FloatContraction,
    /// The comptime arguments, when the kernel has any.
    pub info: Option<Info>,
}

/// Hashes `setting` into `state` once it leaves its default, so a kernel that keeps the default
/// has the hash, entry point name and cache key it had before the setting existed.
fn hash_unless_default<T: Default + PartialEq + Hash, H: Hasher>(setting: &T, state: &mut H) {
    if *setting != T::default() {
        setting.hash(state);
    }
}

/// `-{setting:?}` once `setting` leaves its default, for the same reason as
/// [`hash_unless_default`], and nothing otherwise.
fn suffix_unless_default<T: Default + PartialEq + core::fmt::Debug>(setting: &T) -> String {
    if *setting == T::default() {
        String::new()
    } else {
        format!("-{setting:?}")
    }
}

/// Hashes everything of `id` but its type into `state`, in the order both of its hashes use.
fn hash_settings<H: Hasher>(id: &KernelId, state: &mut H) {
    id.address_type.hash(state);
    id.cube_dim.hash(state);
    id.mode.hash(state);
    id.info.hash(state);
    hash_unless_default(&id.debug_info, state);
    hash_unless_default(&id.float_contraction, state);
}

impl Hash for KernelId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.type_id.hash(state);
        hash_settings(self, state);
    }
}

impl core::fmt::Debug for KernelId {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let mut debug_str = f.debug_struct("KernelId");
        debug_str
            .field("type", &DebugRaw(self.type_name))
            .field("address_type", &self.address_type);
        debug_str.field("cube_dim", &self.cube_dim);
        debug_str.field("mode", &self.mode);
        debug_str.field("debug_info", &self.debug_info);
        debug_str.field("float_contraction", &self.float_contraction);
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
    /// The kernel's type name, trimmed to its last path segment — the short
    /// name a report can print without drowning the reader in type paths.
    pub fn short_name(&self) -> &'static str {
        short_type_name(self.type_name)
    }

    /// The kernel's type, in full: what names it before its comptime
    /// arguments tell instances apart.
    pub fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// The id of a kernel of type `T`, checked, without comptime arguments, on a single unit.
    pub fn new<T: 'static>() -> Self {
        Self {
            type_id: core::any::TypeId::of::<T>(),
            type_name: core::any::type_name::<T>(),
            info: None,
            cube_dim: Dim3::new_single(),
            mode: ExecutionMode::Checked,
            address_type: Default::default(),
            debug_info: DebugInfo::None,
            float_contraction: FloatContraction::Allowed,
        }
    }

    /// The id written the same way in every run: a persistent kernel cache key.
    pub fn stable_format(&self) -> String {
        format!(
            "{}-{}-{:?}-{:?}-{:?}{}{}",
            self.type_name,
            self.address_type,
            self.cube_dim,
            self.mode,
            self.info,
            suffix_unless_default(&self.debug_info),
            suffix_unless_default(&self.float_contraction),
        )
    }

    /// The id hashed the same way in every run: a persistent kernel cache key.
    pub fn stable_hash(&self) -> StableHash {
        let mut hasher = StableHasher::new();
        self.type_name.hash(&mut hasher);
        hash_settings(self, &mut hasher);
        hasher.finalize()
    }

    /// `base` with a discriminator from the [stable hash](Self::stable_hash), so two kernels of
    /// one type get entry points of their own.
    pub fn entrypoint_name(&self, base: &str) -> String {
        format!("{base}_{:08x}", self.stable_hash() as u32)
    }

    /// The same id with `info` as its comptime arguments.
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

    /// The same id with the execution `mode`.
    pub fn mode(mut self, mode: ExecutionMode) -> Self {
        self.mode = mode;
        self
    }

    /// The same id with the debug data the kernel asks for. The id keeps the level after the
    /// configuration lowers it, because that level changes the compiled kernel.
    #[must_use]
    pub fn debug_info(mut self, requested: DebugInfo) -> Self {
        self.debug_info = crate::config::effective_debug_info(requested);
        self
    }

    /// The same id with whether float multiply-adds may be fused, which changes the compiled
    /// kernel.
    #[must_use]
    pub fn float_contraction(mut self, contraction: FloatContraction) -> Self {
        self.float_contraction = contraction;
        self
    }

    /// The same id with the cube dim it is compiled for.
    pub fn cube_dim(mut self, cube_dim: Dim3) -> Self {
        self.cube_dim = cube_dim;
        self
    }

    /// The same id with the type its addresses are computed in.
    pub fn address_type(mut self, addr_ty: AddressType) -> Self {
        self.address_type = addr_ty;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn with_debug_info(debug_info: DebugInfo) -> KernelId {
        KernelId {
            debug_info,
            ..KernelId::new::<()>().info("1")
        }
    }

    #[test]
    fn ids_with_other_info_are_other_keys() {
        let set = HashSet::from([KernelId::new::<()>().info("1")]);

        assert!(!set.contains(&KernelId::new::<()>().info("2")));
    }

    #[test]
    fn debug_info_changes_the_stable_hash() {
        assert_ne!(
            with_debug_info(DebugInfo::None).stable_hash(),
            with_debug_info(DebugInfo::LineTables).stable_hash()
        );
    }

    #[test]
    fn debug_info_changes_the_stable_format() {
        assert_ne!(
            with_debug_info(DebugInfo::LineTables).stable_format(),
            with_debug_info(DebugInfo::Full).stable_format()
        );
    }

    #[test]
    fn debug_info_changes_equality() {
        assert_ne!(
            with_debug_info(DebugInfo::None),
            with_debug_info(DebugInfo::LineTables)
        );
    }

    #[test]
    fn debug_info_changes_the_hash() {
        let state = std::collections::hash_map::RandomState::new();

        assert_ne!(
            core::hash::BuildHasher::hash_one(&state, with_debug_info(DebugInfo::None)),
            core::hash::BuildHasher::hash_one(&state, with_debug_info(DebugInfo::LineTables))
        );
    }

    /// Without debug data, the hash and the entry point name are the same as before the
    /// setting existed.
    #[test]
    fn no_debug_info_keeps_the_stable_hash() {
        assert_eq!(
            with_debug_info(DebugInfo::None).stable_hash(),
            KernelId::new::<()>().info("1").stable_hash()
        );
    }

    #[test]
    fn no_debug_info_keeps_the_stable_format() {
        assert_eq!(
            with_debug_info(DebugInfo::None).stable_format(),
            KernelId::new::<()>().info("1").stable_format()
        );
    }

    #[test]
    fn the_short_name_drops_paths_and_generics() {
        assert_eq!(KernelId::new::<alloc::vec::Vec<u8>>().short_name(), "Vec");
    }

    #[test]
    fn the_entry_point_name_starts_with_its_base() {
        assert!(
            KernelId::new::<()>()
                .entrypoint_name("kernel")
                .starts_with("kernel_")
        );
    }

    #[test]
    fn a_builder_sets_its_setting() {
        assert_eq!(
            KernelId::new::<()>().mode(ExecutionMode::Unchecked).mode,
            ExecutionMode::Unchecked
        );
    }
}
