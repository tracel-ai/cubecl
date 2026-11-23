use super::Info;
use crate::config::short_type_name;
use alloc::format;
use alloc::string::String;
use core::{
    any::TypeId,
    fmt::{Display, Formatter},
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

/// The brackets [`Display`] lays a kernel's comptime arguments out by.
const INFO_BRACKETS: &[(char, char)] = &[('(', ')'), ('[', ']'), ('{', '}')];

/// What [`Display`] writes for a kernel without comptime arguments.
const NO_INFO: &str = "No info";

/// Everything that tells one compiled kernel from another: its type, its launch settings and its
/// comptime arguments.
///
/// Two ids are equal when they compile to the same kernel, and the [stable
/// hash](Self::stable_hash) and [stable format](Self::stable_format) name that kernel the same
/// way in every run, so they key the persistent kernel caches.
///
/// # Examples
///
/// ```
/// use cubecl_runtime::id::KernelId;
/// use cubecl_ir::settings::ExecutionMode;
///
/// struct Add;
///
/// let checked = KernelId::new::<Add>().with_info(4u32);
/// let unchecked = checked.clone().with_mode(ExecutionMode::Unchecked);
///
/// assert_ne!(checked, unchecked);
/// assert_eq!(unchecked.mode(), ExecutionMode::Unchecked);
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct KernelId {
    // Set with `type_id`, from the same type, so equality reads `type_id` alone.
    #[eq(skip)]
    type_name: &'static str,
    type_id: TypeId,
    cube_dim: Dim3,
    address_type: AddressType,
    mode: ExecutionMode,
    debug_info: DebugInfo,
    float_contraction: FloatContraction,
    info: Option<Info>,
}

impl KernelId {
    /// The id of a kernel of type `T`, checked, without comptime arguments, on a single unit.
    pub fn new<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            type_name: core::any::type_name::<T>(),
            info: None,
            cube_dim: Dim3::new_single(),
            mode: ExecutionMode::Checked,
            address_type: Default::default(),
            debug_info: DebugInfo::None,
            float_contraction: FloatContraction::Allowed,
        }
    }

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

    /// The cube dim the kernel is compiled for.
    pub fn cube_dim(&self) -> Dim3 {
        self.cube_dim
    }

    /// The type the kernel computes its addresses in.
    pub fn address_type(&self) -> AddressType {
        self.address_type
    }

    /// Whether the kernel checks its accesses against their bounds.
    pub fn mode(&self) -> ExecutionMode {
        self.mode
    }

    /// The debug data the kernel carries, after the configuration lowered what it asked for.
    pub fn debug_info(&self) -> DebugInfo {
        self.debug_info
    }

    /// Whether the kernel may fuse float multiply-adds.
    pub fn float_contraction(&self) -> FloatContraction {
        self.float_contraction
    }

    /// The id written the same way in every run: a persistent kernel cache key.
    pub fn stable_format(&self) -> String {
        stable_format(self)
    }

    /// The id hashed the same way in every run: a persistent kernel cache key.
    pub fn stable_hash(&self) -> StableHash {
        stable_hash(self)
    }

    /// `base` with a discriminator from the [stable hash](Self::stable_hash), so two kernels of
    /// one type get entry points of their own.
    pub fn entrypoint_name(&self, base: &str) -> String {
        entrypoint_name(self, base)
    }

    /// The same id with `info` as its comptime arguments, which tell apart kernels of one type
    /// that compile to different code.
    pub fn with_info<I: 'static + PartialEq + Eq + Hash + core::fmt::Debug + Send + Sync>(
        self,
        info: I,
    ) -> Self {
        Self {
            info: Some(Info::new(info)),
            ..self
        }
    }

    /// The same id with the execution `mode`.
    pub fn with_mode(self, mode: ExecutionMode) -> Self {
        Self { mode, ..self }
    }

    /// The same id with the debug data the kernel asks for. The id keeps the level after the
    /// configuration lowers it, because that level changes the compiled kernel.
    #[must_use]
    pub fn with_debug_info(self, requested: DebugInfo) -> Self {
        Self {
            debug_info: crate::config::effective_debug_info(requested),
            ..self
        }
    }

    /// The same id with whether float multiply-adds may be fused, which changes the compiled
    /// kernel.
    #[must_use]
    pub fn with_float_contraction(self, float_contraction: FloatContraction) -> Self {
        Self {
            float_contraction,
            ..self
        }
    }

    /// The same id with the cube dim it is compiled for.
    pub fn with_cube_dim(self, cube_dim: Dim3) -> Self {
        Self { cube_dim, ..self }
    }

    /// The same id with the type its addresses are computed in.
    pub fn with_address_type(self, address_type: AddressType) -> Self {
        Self {
            address_type,
            ..self
        }
    }
}

impl Hash for KernelId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_kernel_id(self, state);
    }
}

impl core::fmt::Debug for KernelId {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        debug_kernel_id(self, f)
    }
}

impl Display for KernelId {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        display_kernel_id(self, f)
    }
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

/// The in-process hash: the type by its id, then the settings.
fn hash_kernel_id<H: Hasher>(id: &KernelId, state: &mut H) {
    id.type_id.hash(state);
    hash_settings(id, state);
}

/// The hash across runs: the type by its name, then the settings.
fn stable_hash(id: &KernelId) -> StableHash {
    let mut hasher = StableHasher::new();
    id.type_name.hash(&mut hasher);
    hash_settings(id, &mut hasher);
    hasher.finalize()
}

fn stable_format(id: &KernelId) -> String {
    format!(
        "{}-{}-{:?}-{:?}-{:?}{}{}",
        id.type_name,
        id.address_type,
        id.cube_dim,
        id.mode,
        id.info,
        suffix_unless_default(&id.debug_info),
        suffix_unless_default(&id.float_contraction),
    )
}

fn entrypoint_name(id: &KernelId, base: &str) -> String {
    format!("{base}_{:08x}", stable_hash(id) as u32)
}

fn debug_kernel_id(id: &KernelId, f: &mut Formatter<'_>) -> core::fmt::Result {
    let mut debug_str = f.debug_struct("KernelId");
    debug_str
        .field("type", &DebugRaw(id.type_name))
        .field("address_type", &id.address_type)
        .field("cube_dim", &id.cube_dim)
        .field("mode", &id.mode)
        .field("debug_info", &id.debug_info)
        .field("float_contraction", &id.float_contraction);
    match &id.info {
        Some(info) => debug_str.field("info", info),
        None => debug_str.field("info", &id.info),
    };
    debug_str.finish()
}

fn display_kernel_id(id: &KernelId, f: &mut Formatter<'_>) -> core::fmt::Result {
    match &id.info {
        Some(info) => f.write_str(&format_str(&format!("{info:?}"), INFO_BRACKETS, true)),
        None => f.write_str(NO_INFO),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn with_debug_info(debug_info: DebugInfo) -> KernelId {
        KernelId {
            debug_info,
            ..KernelId::new::<()>().with_info("1")
        }
    }

    #[test]
    fn ids_with_other_info_are_other_keys() {
        let set = HashSet::from([KernelId::new::<()>().with_info("1")]);

        assert!(!set.contains(&KernelId::new::<()>().with_info("2")));
    }

    #[test]
    fn ids_of_other_types_are_not_equal() {
        assert_ne!(KernelId::new::<u8>(), KernelId::new::<u16>());
    }

    /// The persistent caches are keyed by this text: changing it recompiles every kernel.
    #[test]
    fn the_stable_format_keeps_its_layout() {
        assert_eq!(
            KernelId::new::<()>().with_info("1").stable_format(),
            "()-u32-Dim3 { x: 1, y: 1, z: 1 }-Checked-Some(\"1\")"
        );
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
            KernelId::new::<()>().with_info("1").stable_hash()
        );
    }

    #[test]
    fn allowed_float_contraction_keeps_the_stable_hash() {
        assert_eq!(
            KernelId::new::<()>()
                .with_float_contraction(FloatContraction::Allowed)
                .stable_hash(),
            KernelId::new::<()>().stable_hash()
        );
    }

    #[test]
    fn the_short_name_drops_paths_and_generics() {
        assert_eq!(KernelId::new::<alloc::vec::Vec<u8>>().short_name(), "Vec");
    }

    #[test]
    fn the_type_name_is_the_full_path() {
        assert_eq!(
            KernelId::new::<alloc::vec::Vec<u8>>().type_name(),
            "alloc::vec::Vec<u8>"
        );
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
    fn with_mode_sets_the_mode() {
        assert_eq!(
            KernelId::new::<()>()
                .with_mode(ExecutionMode::Unchecked)
                .mode(),
            ExecutionMode::Unchecked
        );
    }

    #[test]
    fn with_cube_dim_sets_the_cube_dim() {
        let cube_dim = Dim3 { x: 8, y: 4, z: 1 };

        assert_eq!(
            KernelId::new::<()>().with_cube_dim(cube_dim).cube_dim(),
            cube_dim
        );
    }

    #[test]
    fn with_address_type_sets_the_address_type() {
        assert_eq!(
            KernelId::new::<()>()
                .with_address_type(AddressType::U64)
                .address_type(),
            AddressType::U64
        );
    }

    #[test]
    fn with_float_contraction_sets_it() {
        assert_eq!(
            KernelId::new::<()>()
                .with_float_contraction(FloatContraction::Off)
                .float_contraction(),
            FloatContraction::Off
        );
    }

    #[test]
    fn display_without_info_says_so() {
        assert_eq!(
            alloc::string::ToString::to_string(&KernelId::new::<()>()),
            NO_INFO
        );
    }

    #[test]
    fn debug_names_the_type() {
        assert!(alloc::format!("{:?}", KernelId::new::<()>()).contains("type: ()"));
    }
}
