use crate::{LineSize, Matrix, MatrixIdent, MatrixLayout, TypeHash};

/// Hacky solution for getting comptime properties into the scope.
/// Allows querying certain target-specific properties at compile time, rather than at runtime.
/// Review on how to better solve this and delegate to the compiler if possible.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash, Default)]
pub struct TargetProperties {
    pub mma: MmaProperties,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
pub struct MmaProperties {
    /// Size of registers in bits, used to calculate line size
    pub register_size_bits: usize,
    /// Constant size of planes, for calculating lane indices in a matrix
    pub const_plane_size: u32,
    /// Layout of registers in Matrix A
    pub register_layout_a: MatrixLayout,
    /// Layout of registers in Matrix B
    pub register_layout_b: MatrixLayout,
    /// Layout of registers in Matrix C/D
    pub register_layout_acc: MatrixLayout,

    /// How many copies of each piece of data exist for matrix A
    pub register_duplication_a: usize,
    /// How many copies of each piece of data exist for matrix B
    pub register_duplication_b: usize,
    /// How many copies of each piece of data exist for matrix C/D
    pub register_duplication_acc: usize,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub contiguous_elements: ContiguousElements,
}

#[derive(Clone)]
pub struct ContiguousElements {
    inner: alloc::rc::Rc<dyn Fn(MatrixIdent, Matrix) -> LineSize>,
}

impl ContiguousElements {
    pub fn new(func: impl Fn(MatrixIdent, Matrix) -> LineSize + 'static) -> Self {
        Self {
            inner: alloc::rc::Rc::new(func),
        }
    }

    pub fn apply(&self, ident: MatrixIdent, matrix: Matrix) -> LineSize {
        (self.inner)(ident, matrix)
    }
}

impl Default for ContiguousElements {
    fn default() -> Self {
        Self {
            inner: alloc::rc::Rc::new(|_, _| 2),
        }
    }
}

impl core::fmt::Debug for ContiguousElements {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ContiguousElements").finish()
    }
}

impl Eq for ContiguousElements {}
impl PartialEq for ContiguousElements {
    fn eq(&self, other: &Self) -> bool {
        alloc::rc::Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl TypeHash for ContiguousElements {
    fn write_hash(hasher: &mut impl core::hash::Hasher) {
        hasher.write_i32(0);
    }
}

impl Default for MmaProperties {
    fn default() -> Self {
        Self {
            register_size_bits: 32,
            const_plane_size: 32,
            register_layout_a: MatrixLayout::RowMajor,
            register_layout_b: MatrixLayout::ColMajor,
            register_layout_acc: MatrixLayout::RowMajor,
            register_duplication_a: 1,
            register_duplication_b: 1,
            register_duplication_acc: 1,
            contiguous_elements: Default::default(),
        }
    }
}
