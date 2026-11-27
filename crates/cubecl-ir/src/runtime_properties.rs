use pliron::{context::Context, r#type::TypedHandle};

use crate::{
    ElemType, ElementPositions, TypeHash, VectorSize,
    types::matrix::{MatrixIdent, MatrixLayout, MatrixScope, MatrixShape, MatrixType},
};

/// What a target states of itself to the kernels expanded for it, read while they expand.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash, Default)]
pub struct TargetProperties {
    pub mma: MmaProperties,
}

/// How a target's manual matrix instruction holds its operands in the registers of a plane.
///
/// # Guarantees
///
/// - A deserialized value holds the defaults of `contiguous_elements` and `element_positions`,
///   which are not serialized.
/// - Two values compare equal only where their operand facts are the same functions: two calls
///   building the same facts give unequal values.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq, TypeHash)]
pub struct MmaProperties {
    /// The bits one register holds, which a vector of an operand's elements fills.
    pub register_size_bits: usize,
    /// The lanes of the plane the instruction runs on, which a lane's place in an operand counts
    /// over.
    pub const_plane_lane_count: u32,
    /// The layout along which a lane's run of A's elements lies.
    pub register_layout_a: MatrixLayout,
    /// The layout along which a lane's run of B's elements lies.
    pub register_layout_b: MatrixLayout,
    /// The layout along which a lane's run of the accumulator's elements lies.
    pub register_layout_acc: MatrixLayout,
    /// The lanes that hold each cell of A.
    pub register_copy_count_a: usize,
    /// The lanes that hold each cell of B.
    pub register_copy_count_b: usize,
    /// The lanes that hold each cell of the accumulator.
    pub register_copy_count_acc: usize,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub contiguous_elements: ContiguousElements,
    /// Where the elements a lane holds of an operand lie in it, where the target has a manual
    /// matrix instruction
    #[cfg_attr(feature = "serde", serde(skip))]
    pub element_positions: OperandFact<Option<ElementPositions>>,
    pub fragment_positions: FragmentPositions,
    /// Whether the compiler's fragment load reads a lane's run of an A or B operand laid across
    /// the rows of its memory more slowly than one laid along them.
    #[cfg_attr(feature = "serde", serde(default))]
    pub strided_run_read: StridedRunRead,
}

/// Whether a compiler's fragment load (`cmma::load`) reads a lane's run of an A or B operand more
/// slowly where the run lies across the rows of the memory it loads from, stepping by the stride,
/// than where it lies along a row.
///
/// # Guarantees
///
/// - It describes the compiler's own fragment load of the fragment API alone, no other load.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, Default)]
pub enum StridedRunRead {
    /// No more slowly: the load reads a strided run at the cost of a contiguous one, whether it
    /// reads both whole or both one element at a time.
    #[default]
    Whole,
    /// One element at a time, where it reads a contiguous run in one access.
    ElementWise,
}

/// Whether a compiler states which lane holds which cell of a fragment of the fragment API.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TypeHash, Default)]
pub enum FragmentPositions {
    /// The compiler hides which lane holds a cell, as `wmma`, rocWMMA, SPIR-V and Metal do.
    #[default]
    Hidden,
    /// A fragment's registers hold their cells where the manual MMA's element positions say.
    AsElementPositions,
}

/// What a target says of each operand of its manual matrix instructions, by the operand's role and
/// its matrix type.
///
/// # Guarantees
///
/// - Two facts compare equal only where they are clones of one fact, and every fact hashes
///   alike, so a hash never tells two facts apart.
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

    /// What the fact says of operand `ident` of type `matrix`.
    ///
    /// # Guarantees
    ///
    /// - The answer is the one the function the fact was built from gives.
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
        Self::with_matrix(ident, shape, elem, |ctx, matrix| {
            self.contiguous_elements.apply(ctx, ident, matrix)
        })
    }

    /// Where the elements a lane holds of operand `ident`, of `elem`, in a contraction of `shape`
    /// lie in it, for code that picks a definition ahead.
    ///
    /// # Assumptions
    ///
    /// - The target's positions depend on neither the operand's layout nor its scope.
    ///
    /// # Guarantees
    ///
    /// - The answer is the positions a kernel reads off its own `MmaDefinition` of that operand,
    ///   `None` where the target has no manual matrix instruction for it.
    pub fn element_positions_of(
        &self,
        ident: MatrixIdent,
        shape: MatrixShape,
        elem: ElemType,
    ) -> Option<ElementPositions> {
        Self::with_matrix(ident, shape, elem, |ctx, matrix| {
            self.element_positions.apply(ctx, ident, matrix)
        })
    }

    /// The elements a lane holds of operand `ident` of `elems` elements, each held by its
    /// operand's register copy count of lanes.
    pub fn elems_per_lane(&self, ident: MatrixIdent, elems: usize) -> usize {
        let copies = match ident {
            MatrixIdent::A => self.register_copy_count_a,
            MatrixIdent::B => self.register_copy_count_b,
            MatrixIdent::Accumulator => self.register_copy_count_acc,
        };
        elems * copies / self.const_plane_lane_count as usize
    }

    /// `fact` of operand `ident`, of `elem`, in a contraction of `shape`, asked in a context of
    /// its own.
    fn with_matrix<T>(
        ident: MatrixIdent,
        shape: MatrixShape,
        elem: ElemType,
        fact: impl FnOnce(&Context, TypedHandle<MatrixType>) -> T,
    ) -> T {
        let ctx = Context::default();
        let matrix = MatrixType::get(
            &ctx,
            ident,
            shape,
            elem.to_type(&ctx),
            MatrixLayout::ColMajor,
            MatrixScope::Plane,
        );
        fact(&ctx, matrix)
    }
}

impl Default for MmaProperties {
    fn default() -> Self {
        Self {
            register_size_bits: 32,
            const_plane_lane_count: 32,
            register_layout_a: MatrixLayout::RowMajor,
            register_layout_b: MatrixLayout::ColMajor,
            register_layout_acc: MatrixLayout::RowMajor,
            register_copy_count_a: 1,
            register_copy_count_b: 1,
            register_copy_count_acc: 1,
            contiguous_elements: Default::default(),
            element_positions: Default::default(),
            fragment_positions: FragmentPositions::default(),
            strided_run_read: StridedRunRead::default(),
        }
    }
}
