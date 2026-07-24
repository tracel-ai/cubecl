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
    /// Create a new cube dim with x = y = z = 1.
    pub const fn new_single() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new cube dim with the given x, y and z.
    /// This is equivalent to the [new](CubeDim::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total numbers of units per cube
    pub const fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Whether this `CubeDim` can fully contain `other`
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSettings {
    /// The cube dim of the kernel
    pub cube_dim: Dim3,
    /// The address type of the kernel
    pub address_type: AddressType,
    /// The name of the kernel
    pub kernel_name: String,
    /// Whether to include debug symbols
    pub debug_symbols: bool,
    /// CUDA Cluster dim, if any
    pub cluster_dim: Option<Dim3>,
    /// Execution mode
    pub execution_mode: ExecutionMode,
}

impl KernelSettings {
    pub fn new(cube_dim: Dim3, execution_mode: ExecutionMode, address_type: AddressType) -> Self {
        Self {
            cube_dim,
            address_type,
            kernel_name: String::new(),
            debug_symbols: false,
            cluster_dim: None,
            execution_mode,
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

    /// Activate debug symbols
    pub fn debug_symbols(mut self) -> Self {
        self.debug_symbols = true;
        self
    }

    /// Set cluster dim
    pub fn cluster_dim(mut self, cluster_dim: Dim3) -> Self {
        self.cluster_dim = Some(cluster_dim);
        self
    }
}
