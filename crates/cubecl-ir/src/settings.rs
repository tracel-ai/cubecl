use alloc::string::{String, ToString};
use pliron::derive::format;

use crate::AddressType;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, serde::Serialize, serde::Deserialize)]
#[allow(missing_docs)]
#[format("`(` $x `, ` $y `, ` $z `)`")]
/// The number of units across all 3 axis totalling to the number of working units in a cube.
pub struct Dim3 {
    /// The number of units in the x axis.
    pub x: u32,
    /// The number of units in the y axis.
    pub y: u32,
    /// The number of units in the z axis.
    pub z: u32,
}

impl Dim3 {
    /// Create a new `Dim3` with x = y = z = 1.
    pub const fn new_single() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    /// Create a new `Dim3` with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new dim3 with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new `Dim3` with the given x, y and z.
    /// This is equivalent to the [new](Dim3::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total numbers of units per cube
    pub const fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Whether this `Dim3` can fully contain `other`
    pub const fn can_contain(&self, other: Dim3) -> bool {
        self.x >= other.x && self.y >= other.y && self.z >= other.z
    }
}

impl From<(u32, u32, u32)> for Dim3 {
    fn from(value: (u32, u32, u32)) -> Self {
        Dim3::new_3d(value.0, value.1, value.2)
    }
}

impl From<Dim3> for (u32, u32, u32) {
    fn from(val: Dim3) -> Self {
        (val.x, val.y, val.z)
    }
}

/// The kind of execution to be performed.
#[derive(
    Default, Hash, PartialEq, Eq, Clone, Debug, Copy, serde::Serialize, serde::Deserialize,
)]
pub enum ExecutionMode {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Validate OOB and alert if OOB access occurs
    Validate,
    /// Unchecked kernels are unsafe.
    Unchecked,
}

/// How much source-level debug data a kernel carries. Profilers and debuggers read it to map
/// kernel code back to the `#[cube]` source.
#[derive(
    Default,
    Hash,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Clone,
    Copy,
    Debug,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum DebugInfo {
    /// No debug data.
    #[default]
    #[serde(rename = "none")]
    None,
    /// Source lines, with each inlined `#[cube]` function as its own frame.
    #[serde(rename = "line-tables")]
    LineTables,
    /// Line tables and the embedded source text.
    #[serde(rename = "full")]
    Full,
}

/// Whether the compiler may contract a float multiply and the add or subtract that consumes it
/// into one fused multiply-add, rounding once instead of twice.
///
/// Contraction is allowed by default on every backend. A kernel that must give the same bits
/// as an uncontracted reference, on every backend and vector size, turns it off with
/// `#[cube(no_contraction)]` or [`KernelSettings::no_contraction`]. Explicit `fma` calls are
/// fused either way.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum FloatContraction {
    /// Products may fuse with the add or subtract that consumes them.
    #[default]
    Allowed,
    /// Every product and every sum is rounded on its own.
    Off,
}

/// The buffer bindings of a launch that every byte offset into fits in 32 bits, by position.
///
/// A backend that knows a binding is this small can address it with a 32-bit byte offset from a
/// base shared by the whole launch, rather than a 64-bit address per access: on AMD GPUs, a load
/// with its base in scalar registers. A launch marks a binding only when the device reports
/// [`Features::narrow_bindings`](crate::Features::narrow_bindings), so on every other device the
/// set is empty and the kernel compiles as if it did not exist.
///
/// # Guarantees
///
/// - Holds only positions below 64; a binding past them is never marked, which is always safe.
/// - The empty set is the default, so a kernel with no marked binding keeps its id.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct NarrowBindings {
    /// Bit `p` set when binding `p` is marked.
    positions: u64,
}

impl NarrowBindings {
    /// The most bytes a binding may hold to be marked: every byte offset into it is below 2^32.
    pub const MAX_BYTES: u64 = 1 << 32;

    /// Whether a binding of `size_bytes` bytes may be marked.
    pub fn fits_size_bytes(size_bytes: u64) -> bool {
        size_bytes <= Self::MAX_BYTES
    }

    /// Marks binding `position`; a position of 64 or more is left unmarked.
    pub fn insert_position(&mut self, position: usize) {
        self.positions = with_position(self.positions, position);
    }

    /// Whether binding `position` is marked.
    pub fn contains_position(&self, position: usize) -> bool {
        holds_position(self.positions, position)
    }
}

/// The bit of binding `position` in a [`NarrowBindings`], `None` past the 64 it holds.
fn position_bit(position: usize) -> Option<u64> {
    u32::try_from(position)
        .ok()
        .and_then(|position| 1u64.checked_shl(position))
}

/// `positions` with binding `position`'s bit set, unchanged past the 64 it holds.
fn with_position(positions: u64, position: usize) -> u64 {
    positions | position_bit(position).unwrap_or(0)
}

/// Whether `positions` has binding `position`'s bit set.
fn holds_position(positions: u64, position: usize) -> bool {
    position_bit(position).is_some_and(|bit| positions & bit != 0)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelSettings {
    /// The cube dim of the kernel
    pub cube_dim: Dim3,
    /// The address type of the kernel
    pub address_type: AddressType,
    /// The name of the kernel
    pub kernel_name: String,
    /// The source-level debug data to include
    pub debug_info: DebugInfo,
    /// CUDA Cluster dim, if any
    pub cluster_dim: Option<Dim3>,
    /// Execution mode
    pub execution_mode: ExecutionMode,
    /// Whether float multiply-adds may be fused.
    pub float_contraction: FloatContraction,
    /// The bindings small enough to address with 32-bit byte offsets.
    pub narrow_bindings: NarrowBindings,
}

impl KernelSettings {
    pub fn new(cube_dim: Dim3, execution_mode: ExecutionMode, address_type: AddressType) -> Self {
        Self {
            cube_dim,
            address_type,
            kernel_name: String::new(),
            debug_info: DebugInfo::None,
            cluster_dim: None,
            execution_mode,
            float_contraction: FloatContraction::Allowed,
            narrow_bindings: NarrowBindings::default(),
        }
    }
}

impl KernelSettings {
    /// Set cube dimension.
    pub fn cube_dim(mut self, cube_dim: Dim3) -> Self {
        self.cube_dim = cube_dim;
        self
    }

    /// Set address type.
    pub fn address_type(mut self, ty: AddressType) -> Self {
        self.address_type = ty;
        self
    }

    /// Set kernel name.
    pub fn kernel_name<S: AsRef<str>>(mut self, name: S) -> Self {
        self.kernel_name = name.as_ref().to_string();
        self
    }

    /// Include full debug data, whatever the cargo profile asks for.
    #[must_use]
    pub fn debug_symbols(self) -> Self {
        self.debug_info(DebugInfo::Full)
    }

    /// Set the source-level debug data to include.
    #[must_use]
    pub fn debug_info(mut self, debug_info: DebugInfo) -> Self {
        self.debug_info = debug_info;
        self
    }

    /// Set cluster dim
    pub fn cluster_dim(mut self, cluster_dim: Dim3) -> Self {
        self.cluster_dim = Some(cluster_dim);
        self
    }

    /// Round every float product and sum on its own: no multiply-add is fused unless the
    /// kernel calls `fma`.
    #[must_use]
    pub fn no_contraction(mut self) -> Self {
        self.float_contraction = FloatContraction::Off;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_binding_of_four_gib_fits() {
        assert!(NarrowBindings::fits_size_bytes(0));
        assert!(NarrowBindings::fits_size_bytes(NarrowBindings::MAX_BYTES));
    }

    #[test]
    fn a_binding_past_four_gib_does_not_fit() {
        assert!(!NarrowBindings::fits_size_bytes(
            NarrowBindings::MAX_BYTES + 1
        ));
    }

    #[test]
    fn a_marked_binding_is_contained_alone() {
        let mut bindings = NarrowBindings::default();
        bindings.insert_position(3);
        assert!(bindings.contains_position(3));
        assert!(!bindings.contains_position(2));
    }

    #[test]
    fn a_position_past_64_is_never_narrow() {
        let mut bindings = NarrowBindings::default();
        bindings.insert_position(64);
        assert!(!bindings.contains_position(64));
        assert_eq!(bindings, NarrowBindings::default());
    }
}
