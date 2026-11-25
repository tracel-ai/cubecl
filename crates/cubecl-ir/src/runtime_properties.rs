use pliron::{context::Context, r#type::TypedHandle};

use crate::{
    ElemType, ElementPositions, TypeHash, VectorSize,
    types::matrix::{MatrixIdent, MatrixLayout, MatrixScope, MatrixShape, MatrixType},
};

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
    /// Size of registers in bits, used to calculate vector size
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
    /// Where the elements a lane holds of an operand lie in it, where the target has a manual
    /// matrix instruction
    #[cfg_attr(feature = "serde", serde(skip))]
    pub element_positions: OperandFact<Option<ElementPositions>>,
}

/// What a target says of each operand of its manual matrix instructions, by the operand's role and
/// its matrix type.
#[derive(Clone)]
pub struct OperandFact<T> {
    #[allow(clippy::type_complexity)]
    inner:
        alloc::sync::Arc<dyn Fn(&Context, MatrixIdent, TypedHandle<MatrixType>) -> T + Send + Sync>,
}

/// The consecutive elements of an operand a lane holds in one vector.
pub type ContiguousElements = OperandFact<VectorSize>;

impl<T> OperandFact<T> {
    pub fn new(
        func: impl Fn(&Context, MatrixIdent, TypedHandle<MatrixType>) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner: alloc::sync::Arc::new(func),
        }
    }

    pub fn apply(&self, ctx: &Context, ident: MatrixIdent, matrix: TypedHandle<MatrixType>) -> T {
        (self.inner)(ctx, ident, matrix)
    }
}

impl Default for OperandFact<VectorSize> {
    fn default() -> Self {
        Self::new(|_, _, _| 2)
    }
}

impl Default for OperandFact<Option<ElementPositions>> {
    fn default() -> Self {
        Self::new(|_, _, _| None)
    }
}

impl<T> core::fmt::Debug for OperandFact<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("OperandFact").finish()
    }
}

impl<T> Eq for OperandFact<T> {}
impl<T> PartialEq for OperandFact<T> {
    fn eq(&self, other: &Self) -> bool {
        alloc::sync::Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T> TypeHash for OperandFact<T> {
    fn write_hash(hasher: &mut impl core::hash::Hasher) {
        hasher.write_i32(0);
    }
}

impl MmaProperties {
    /// Elements a lane holds in one vector of operand `ident`, of `elem`, in a contraction of
    /// `shape`: what a kernel reads off its own definition of it, for code that picks one ahead.
    pub fn vector_size(
        &self,
        ident: MatrixIdent,
        shape: MatrixShape,
        elem: ElemType,
    ) -> VectorSize {
        let ctx = Context::default();
        let matrix = MatrixType::get(
            &ctx,
            ident,
            shape,
            elem.to_type(&ctx),
            MatrixLayout::ColMajor,
            MatrixScope::Plane,
        );
        self.contiguous_elements.apply(&ctx, ident, matrix)
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
            element_positions: Default::default(),
        }
    }
}
