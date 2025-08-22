use serde::{Deserialize, Serialize};

/// An approximation of the plane dimension.
pub const PLANE_DIM_APPROX: usize = 16;

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(new, Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[allow(missing_docs)]
pub struct CubeDim {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl CubeDim {
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
    pub const fn can_contain(&self, other: CubeDim) -> bool {
        self.x >= other.x && self.y >= other.y && self.z >= other.z
    }
}

impl Default for CubeDim {
    fn default() -> Self {
        Self {
            x: PLANE_DIM_APPROX as u32,
            y: PLANE_DIM_APPROX as u32,
            z: 1,
        }
    }
}

/// The kind of execution to be performed.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Unchecked kernels are unsafe.
    Unchecked,
}
