//! This module exposes cooperative matrix-multiply and accumulate operations.
//!
//! Most of the functions are actually unsafe, since they mutate their input, even if they are
//! passed as reference.
//!
//! # Example
//!
//! This is a basic 16x16x16 matrix multiplication example.
//!
//! ```rust, ignore
//! #[cube(launch)]
//! pub fn example(lhs: &[f16], rhs: &[f16], out: &mut [f32]) {
//!     let a = cmma::Matrix::<f16>::new(
//!         cmma::MatrixIdent::A,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::RowMajor,
//!     );
//!     let b = cmma::Matrix::<f16>::new(
//!         cmma::MatrixIdent::B,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::ColMajor,
//!     );
//!     let c = cmma::Matrix::<f32>::new(
//!         cmma::MatrixIdent::Accumulator,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::Undefined,
//!     );
//!     cmma::fill(&c, 0.0);
//!     cmma::load(&a, lhs.as_slice(), 16);
//!     cmma::load(&b, rhs.as_slice(), 16);
//!
//!     cmma::execute(&a, &b, &c, &c);
//!
//!     cmma::store(
//!         out.as_mut_slice(),
//!         &c,
//!         16,
//!         cmma::MatrixLayout::RowMajor,
//!     );
//! }
//! ```

use super::{CubeDebug, CubePrimitive, CubeType, IntoMut, NativeExpand, SliceExpand};
use crate::{self as cubecl, prelude::*};
use crate::{
    ir::{self, Instruction},
    unexpanded,
};
use core::marker::PhantomData;
use cubecl_macros::{comptime_type, cube, intrinsic};

use cubecl_ir::{CoopMma, Scope, StorageType, Variable, VectorSize};
pub use ir::{MatrixIdent, MatrixLayout};

#[derive(Clone, Copy)]
pub struct Plane;
#[derive(Clone, Copy)]
pub struct Cube;

pub trait MatrixScope: Copy {
    const SCOPE: ir::MatrixScope;
}

impl MatrixScope for Plane {
    const SCOPE: ir::MatrixScope = ir::MatrixScope::Plane;
}

impl MatrixScope for Cube {
    const SCOPE: cubecl_ir::MatrixScope = ir::MatrixScope::Cube;
}

/// A matrix represent a 2D grid of numbers.
///
/// They can either be in a [row major](MatrixLayout::RowMajor) or a
/// [column major](MatrixLayout::ColMajor) format.
#[derive(Copy, Clone)]
pub struct Matrix<C: CubeType, S: MatrixScope = Plane> {
    _c: PhantomData<C>,
    _s: PhantomData<S>,
}

/// Defines a matrix multiplication operation, including the input and output type, and the shape.
#[derive(Copy, Clone)]
pub struct MmaDefinition<A: CubeType, B: CubeType, CD: CubeType> {
    _a: PhantomData<A>,
    _b: PhantomData<B>,
    _cd: PhantomData<CD>,
}

/// Expand type of [Matrix].
pub struct MatrixExpand<C: CubeType, S: MatrixScope> {
    elem: Variable,
    ident: MatrixIdent,
    _c: PhantomData<C>,
    _s: PhantomData<S>,
}

/// Expand type of [`MmaDefinition`].
#[derive(Debug)]
pub struct MmaDefinitionExpand<A: CubeType, B: CubeType, CD: CubeType> {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub a_type: StorageType,
    pub b_type: StorageType,
    pub cd_type: StorageType,
    pub scales_factor: Option<usize>,
    pub scales_type: Option<StorageType>,
    _a: PhantomData<A>,
    _b: PhantomData<B>,
    _cd: PhantomData<CD>,
}

impl<C: CubeType, S: MatrixScope> Clone for MatrixExpand<C, S> {
    fn clone(&self) -> Self {
        Self {
            elem: self.elem,
            ident: self.ident,
            _c: self._c,
            _s: self._s,
        }
    }
}

impl<C: CubeType, S: MatrixScope> ExpandTypeClone for MatrixExpand<C, S> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> ExpandTypeClone for MmaDefinitionExpand<A, B, CD> {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

impl<C: CubeType, S: MatrixScope> AsRefExpand for MatrixExpand<C, S> {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}
impl<C: CubeType, S: MatrixScope> AsMutExpand for MatrixExpand<C, S> {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> AsRefExpand for MmaDefinitionExpand<A, B, CD> {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}
impl<A: CubeType, B: CubeType, CD: CubeType> AsMutExpand for MmaDefinitionExpand<A, B, CD> {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> Copy for MmaDefinitionExpand<A, B, CD> {}
impl<A: CubeType, B: CubeType, CD: CubeType> Clone for MmaDefinitionExpand<A, B, CD> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: CubeType, S: MatrixScope> CubeType for Matrix<C, S> {
    type ExpandType = MatrixExpand<C, S>;
}

impl<A: CubeType, B: CubeType, CD: CubeType> CubeType for MmaDefinition<A, B, CD> {
    type ExpandType = MmaDefinitionExpand<A, B, CD>;
}

impl<C: CubeType, S: MatrixScope> IntoExpand for MatrixExpand<C, S> {
    type Expand = Self;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<C: CubeType, S: MatrixScope> IntoMut for MatrixExpand<C, S> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<C: CubeType, S: MatrixScope> CubeDebug for MatrixExpand<C, S> {
    fn set_debug_name(&self, scope: &Scope, name: &'static str) {
        scope.update_variable_name(self.elem, name);
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> IntoExpand for MmaDefinitionExpand<A, B, CD> {
    type Expand = Self;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> IntoMut for MmaDefinitionExpand<A, B, CD> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> CubeDebug for MmaDefinitionExpand<A, B, CD> {}

#[cube]
impl<C: CubePrimitive, S: MatrixScope> Matrix<C, S> {
    /// Create a new uninitialized matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function.
    ///
    /// # Safety
    /// Must be initialized with `load` or `fill` before use. Using it without initialization is
    /// undefined behaviour on CUDA, and completely invalid on Vulkan.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [`MatrixIdent::B`] Shape => (K, N)
    /// * [`MatrixIdent::Accumulator`] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub unsafe fn uninitialized(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        layout: MatrixLayout,
    ) -> Self {
        intrinsic!(|scope| {
            let elem = C::__expand_as_type(scope).storage_type();
            let elem = scope.create_matrix(ir::Matrix::new(ident, m, n, k, elem, layout, S::SCOPE));
            MatrixExpand {
                elem,
                ident,
                _c: PhantomData,
                _s: PhantomData,
            }
        })
    }

    /// Create a new matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function and is filled with `value`.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [`MatrixIdent::B`] Shape => (K, N)
    /// * [`MatrixIdent::Accumulator`] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn from_value(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        layout: MatrixLayout,
        value: C,
    ) -> Self
    where
        C: Scalar,
    {
        let mut mat = unsafe { Self::uninitialized(ident, m, n, k, layout) };
        fill(&mut mat, value);
        mat
    }

    /// Create a new matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function and is loaded from `value` with `stride`.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [`MatrixIdent::B`] Shape => (K, N)
    /// * [`MatrixIdent::Accumulator`] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn from_slice(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        layout: MatrixLayout,
        value: &[C],
        stride: u32,
    ) -> Self {
        let mut mat = unsafe { Self::uninitialized(ident, m, n, k, layout) };

        if comptime![ident == MatrixIdent::Accumulator] {
            load_with_layout(&mut mat, value, stride, layout);
        } else {
            load(&mut mat, value, stride);
        }
        mat
    }

    /// Create a new matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function and is loaded from `value` with `stride`.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [`MatrixIdent::B`] Shape => (K, N)
    /// * [`MatrixIdent::Accumulator`] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn from_tensor(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        value: &TensorView<C>,
    ) -> Self {
        let mut mat = unsafe { Self::uninitialized(ident, m, n, k, MatrixLayout::Undefined) };
        load_tensor(&mut mat, value);
        mat
    }
}

#[cube]
impl<A: Scalar, B: Scalar, CD: Scalar> MmaDefinition<A, B, CD> {
    /// Create a new matrix definition that is going to be used in the manual
    /// matrix-multiply and accumulate ``execute_manual_mma()`` function.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [`MatrixIdent::B`] Shape => (K, N)
    /// * [`MatrixIdent::Accumulator`] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    /// Layout for manual MMA is determined by the runtime and must be handled manually.
    /// Use [`Self::vector_layout`] to check the correct data layout for each element.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn new(#[comptime] m: usize, #[comptime] n: usize, #[comptime] k: usize) -> Self {
        intrinsic!(|scope| {
            let a_type = A::__expand_as_type(scope).storage_type();
            let b_type = B::__expand_as_type(scope).storage_type();
            let cd_type = CD::__expand_as_type(scope).storage_type();

            MmaDefinitionExpand {
                m,
                n,
                k,
                a_type,
                b_type,
                cd_type,
                scales_factor: None,
                scales_type: None,
                _a: PhantomData,
                _b: PhantomData,
                _cd: PhantomData,
            }
        })
    }

    /// Create a new matrix definition that is going to be used in the manual
    /// matrix-multiply and accumulate ``execute_manual_mma()`` function.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [`MatrixIdent::B`] Shape => (K, N)
    /// * [`MatrixIdent::Accumulator`] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    /// Layout for manual MMA is determined by the runtime and must be handled manually.
    /// Use [`Self::vector_layout`] to check the correct data layout for each element.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn new_scaled<S: CubePrimitive>(
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] scale_factor: usize,
    ) -> Self {
        intrinsic!(|scope| {
            let a_type = A::__expand_as_type(scope).storage_type();
            let b_type = B::__expand_as_type(scope).storage_type();
            let cd_type = CD::__expand_as_type(scope).storage_type();

            MmaDefinitionExpand {
                m,
                n,
                k,
                a_type,
                b_type,
                cd_type,
                scales_factor: Some(scale_factor),
                scales_type: Some(S::__expand_as_type(scope).storage_type()),
                _a: PhantomData,
                _b: PhantomData,
                _cd: PhantomData,
            }
        })
    }

    /// Number of elements in the matrix
    #[allow(unused)]
    pub fn num_elems(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            match ident {
                MatrixIdent::A => (self.m * self.k) / self.a_type.packing_factor(),
                MatrixIdent::B => (self.k * self.n) / self.b_type.packing_factor(),
                MatrixIdent::Accumulator => (self.m * self.n) / self.cd_type.packing_factor(),
            }
        })
    }

    /// Returns the number of elements handled by each lane. Should be packed into `Vector`s of size
    /// `vector_size` with [`Self::vector_layout`].
    ///
    /// # Note
    /// "Lane" here refers to the unit relative to a plane, to distinguish it from a unit relative
    /// to a cube.
    #[allow(unused)]
    pub fn elems_per_lane(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            let elems = self.__expand_num_elems_method(scope, ident);
            let plane_dim = scope.state().target_properties.mma.const_plane_size as usize;
            let duplication = match ident {
                MatrixIdent::A => scope.state().target_properties.mma.register_duplication_a,
                MatrixIdent::B => scope.state().target_properties.mma.register_duplication_b,
                MatrixIdent::Accumulator => {
                    scope.state().target_properties.mma.register_duplication_acc
                }
            };
            (elems * duplication) / plane_dim
        })
    }

    /// Returns the number of vectors of size `vector_size` with layout `vector_layout` per lane.
    ///
    /// # Note
    /// "Lane" here refers to the unit relative to a plane, to distinguish it from a unit relative
    /// to a cube.
    #[allow(unused)]
    pub fn vectors_per_lane(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            let elems = self.clone().__expand_elems_per_lane_method(scope, ident);
            let vector_size = self.__expand_vector_size_method(scope, ident);
            elems / vector_size
        })
    }

    /// The layout of each vector in this matrix (row major or column major)
    #[allow(unused)]
    pub fn vector_layout(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(MatrixLayout) {
        intrinsic!(|scope| {
            match ident {
                MatrixIdent::A => scope.state().target_properties.mma.register_layout_a,
                MatrixIdent::B => scope.state().target_properties.mma.register_layout_b,
                MatrixIdent::Accumulator => scope.state().target_properties.mma.register_layout_acc,
            }
        })
    }

    /// Number of elements in each vector passed to the execute function. Represents the maximum
    /// number of contiguous elements held by the thread.
    #[allow(unused_variables)]
    pub fn vector_size(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(VectorSize) {
        intrinsic!(|scope| {
            let storage = match ident {
                MatrixIdent::A => self.a_type,
                MatrixIdent::B => self.b_type,
                MatrixIdent::Accumulator => self.cd_type,
            };
            let matrix = cubecl_ir::Matrix {
                ident,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: storage,
                layout: MatrixLayout::ColMajor,
                scope: ir::MatrixScope::Plane,
            };
            scope
                .state()
                .target_properties
                .mma
                .contiguous_elements
                .apply(ident, matrix)
        })
    }

    /// Returns the coordinates of the `nth` element handled by the `lane_id`
    /// Each lane contains [`Self::elems_per_lane`] elements in [`Self::vector_size`] chunks.
    /// Returns (`row_idx`, `col_idx`)
    ///
    /// # Note
    /// "Lane" here refers to the unit relative to a plane, to distinguish it from a unit relative
    /// to a cube.
    #[allow(unused_variables)]
    pub fn position_of_nth(
        &self,
        lane_id: u32,
        elem_idx: u32,
        #[comptime] ident: MatrixIdent,
    ) -> (u32, u32) {
        intrinsic!(|scope| {
            let lane_id: Variable = lane_id.into();
            let elem_idx: Variable = elem_idx.into();

            let ty = match ident {
                MatrixIdent::A => self.a_type,
                MatrixIdent::B => self.b_type,
                MatrixIdent::Accumulator => self.cd_type,
            };
            let layout = match ident {
                MatrixIdent::A => scope.state().target_properties.mma.register_layout_a,
                MatrixIdent::B => scope.state().target_properties.mma.register_layout_b,
                MatrixIdent::Accumulator => scope.state().target_properties.mma.register_layout_acc,
            };
            let matrix = cubecl_ir::Matrix {
                ident,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: ty,
                layout,
                scope: ir::MatrixScope::Plane,
            };

            let row = scope.create_local(u32::__expand_as_type(scope));
            let col = scope.create_local(u32::__expand_as_type(scope));
            scope.register(Instruction::new(
                CoopMma::RowIndex {
                    lane_id,
                    i: elem_idx,
                    matrix,
                },
                row,
            ));
            scope.register(Instruction::new(
                CoopMma::ColIndex {
                    lane_id,
                    i: elem_idx,
                    matrix,
                },
                col,
            ));
            (row.into(), col.into())
        })
    }

    /// Index of the scales for this thread, along the non-major dimension of the matrix.
    /// Each thread loads all scales in the major direction into a single `Vector`.
    pub fn scales_index(&self, lane_id: u32, #[comptime] ident: MatrixIdent) -> u32 {
        // Just do CUDA for now, call an actual intrinsic when HIP gets support
        let quad_id = lane_id / 4;
        let t_id = lane_id % 4;
        match ident {
            MatrixIdent::A => quad_id + (t_id % 2) * 8,
            MatrixIdent::B => quad_id,
            MatrixIdent::Accumulator => panic!("Accumulator doesn't have scales"),
        }
    }

    /// Number of scales in each vector (not the vector size!). Vector size may include padding bytes.
    pub fn scales_count(&self) -> comptime_type!(usize) {
        // We only have the CUDA version for now, so just use `scales_factor`. The function can
        // be modified for HIP in the future without having to redo all uses.
        intrinsic!(|_| {
            self.scales_factor
                .expect("Can't retrieve scales count for matrix with no scales")
        })
    }

    /// Vector size for the scale factors. May be larger than the total number of scales.
    pub fn scales_vector_size(&self) -> comptime_type!(VectorSize) {
        intrinsic!(|scope| {
            let elem = self
                .scales_type
                .expect("Can't retrieve scales vector size for matrix with no scales");
            scope.state().target_properties.mma.register_size_bits / elem.size_bits()
        })
    }

    /// Load one or more matrix register using intrinsic instructions. CUDA only.
    /// The number of matrices must be 1, 2, or 4. The rows for the nth matrix are passed by the 8
    /// lanes starting at `n * 8`. All slice starts must be valid, even for non-participating lanes.
    /// The slice determines the starting address for a 16-byte row loaded by this unit, with
    /// the row index being `UNIT_POS_PLANE % 8`.
    /// The number of elements is determined by element size.
    ///
    /// # Constraints:
    /// Address must be aligned to 16 bytes
    /// Address must be in shared memory
    #[allow(unused_variables)]
    pub fn load_matrix<E: CubePrimitive, NO: Size>(
        &self,
        row: &[E],
        #[comptime] ident: MatrixIdent,
        #[comptime] num_matrices: usize,
        #[comptime] transpose: bool,
    ) -> Array<Vector<E::Scalar, NO>> {
        intrinsic!(|scope| {
            let slice_vector_size = row.expand.vector_size();
            let ptr = unsafe { *row.__expand_as_ptr_method(scope) }.expand;
            let out = Array::__expand_new(scope, num_matrices);
            scope.register(Instruction::new(
                CoopMma::LoadMatrix {
                    ptr,
                    factor: num_matrices,
                    transpose,
                },
                out.__extract_list(scope),
            ));
            out
        })
    }

    #[allow(unused_variables)]
    pub fn load_matrix_inplace<E: Scalar, N: Size>(
        &self,
        row: &[E],
        fragment: &mut Array<Vector<E, N>>,
        #[comptime] ident: MatrixIdent,
        #[comptime] num_matrices: usize,
        #[comptime] transpose: bool,
    ) {
        intrinsic!(|scope| {
            let vector_size = self.__expand_vector_size_method(scope, ident);
            let slice_vector_size = row.expand.vector_size();
            let ptr = unsafe { *row.__expand_as_ptr_method(scope) }.expand;
            let fragment = fragment.__extract_list(scope);
            scope.register(Instruction::new(
                CoopMma::LoadMatrix {
                    ptr,
                    factor: num_matrices,
                    transpose,
                },
                fragment,
            ));
        })
    }

    /// Store one or more matrix register using intrinsic instructions. CUDA only.
    /// The number of matrices must be 1, 2, or 4. The rows for the nth matrix are passed by the 8
    /// lanes starting at `n * 8`. All slice starts must be valid, even for non-participating lanes.
    /// The slice determines the starting address for a 16-byte row loaded by this unit, with
    /// the row index being `UNIT_POS_PLANE % 8`.
    /// The number of elements is determined by element size.
    ///
    /// # Constraints:
    /// Address must be aligned to 16 bytes
    /// Address must be in shared memory
    #[allow(unused_variables)]
    pub fn store_matrix<E: CubePrimitive, N: Size>(
        &self,
        row: &mut [E],
        registers: &Array<Vector<E::Scalar, N>>,
        #[comptime] ident: MatrixIdent,
        #[comptime] num_matrices: usize,
        #[comptime] transpose: bool,
    ) {
        intrinsic!(|scope| {
            let vector_size = self.__expand_vector_size_method(scope, ident);

            let registers = registers.__extract_list(scope);
            let destination = unsafe { *row.__expand_as_ptr_method(scope) }.expand;

            scope.register(Instruction::no_out(CoopMma::StoreMatrix {
                registers,
                destination,
                factor: num_matrices,
                transpose,
            }));
        })
    }

    /// Execute a low level `mma` operation with manually managed registers. Register layout
    /// and index mapping can be retrieved from the [`MmaDefinition`]
    #[allow(unused)]
    pub fn execute<NA: Size, NB: Size, NC: Size>(
        &self,
        registers_a: &Array<Vector<A, NA>>,
        registers_b: &Array<Vector<B, NB>>,
        registers_c: &Array<Vector<CD, NC>>,
    ) -> Array<Vector<CD, NC>> {
        intrinsic!(|scope| {
            let acc_elems = self
                .clone()
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let acc_vector_size = self
                .clone()
                .__expand_vector_size_method(scope, MatrixIdent::Accumulator);
            let num_registers = acc_elems / acc_vector_size;

            let registers_d = Array::__expand_new(scope, num_registers);

            let registers_a = registers_a.__extract_list(scope);
            let registers_b = registers_b.__extract_list(scope);
            let registers_c = registers_c.__extract_list(scope);

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: self.a_type,
                layout: MatrixLayout::ColMajor,
                scope: ir::MatrixScope::Plane,
            };

            scope.register(Instruction::new(
                CoopMma::ExecuteManual {
                    matrix,
                    registers_a,
                    registers_b,
                    registers_c,
                },
                registers_d.__extract_list(scope),
            ));

            registers_d
        })
    }

    #[allow(unused)]
    pub fn execute_inplace<NA: Size, NB: Size, NC: Size>(
        &self,
        registers_a: &Array<Vector<A, NA>>,
        registers_b: &Array<Vector<B, NB>>,
        registers_c: &mut Array<Vector<CD, NC>>,
    ) {
        intrinsic!(|scope| {
            let acc_elems = self
                .clone()
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let acc_vector_size = self
                .clone()
                .__expand_vector_size_method(scope, MatrixIdent::Accumulator);
            let num_registers = acc_elems / acc_vector_size;

            let registers_a = registers_a.__extract_list(scope);
            let registers_b = registers_b.__extract_list(scope);
            let registers_c = registers_c.__extract_list(scope);

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: self.a_type,
                layout: MatrixLayout::ColMajor,
                scope: ir::MatrixScope::Plane,
            };

            scope.register(Instruction::new(
                CoopMma::ExecuteManual {
                    matrix,
                    registers_a,
                    registers_b,
                    registers_c,
                },
                registers_c,
            ));
        })
    }

    /// Execute a low level block scaled `mma` operation with manually managed registers. Register
    /// layout and index mapping can be retrieved from the [`MmaDefinition`]
    #[allow(unused)]
    pub fn execute_scaled<S: Scalar, NA: Size, NB: Size, NC: Size, NS: Size>(
        &self,
        registers_a: &Array<Vector<A, NA>>,
        registers_b: &Array<Vector<B, NB>>,
        registers_c: &Array<Vector<CD, NC>>,
        scales_a: Vector<S, NS>,
        scales_b: Vector<S, NS>,
    ) -> Array<Vector<CD, NC>> {
        intrinsic!(|scope| {
            let acc_elems = self
                .clone()
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let acc_vector_size = self
                .clone()
                .__expand_vector_size_method(scope, MatrixIdent::Accumulator);
            let num_registers = acc_elems / acc_vector_size;

            let registers_d = Array::__expand_new(scope, num_registers);

            let registers_a = registers_a.__extract_list(scope);
            let registers_b = registers_b.__extract_list(scope);
            let registers_c = registers_c.__extract_list(scope);

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: self.a_type,
                layout: MatrixLayout::ColMajor,
                scope: ir::MatrixScope::Plane,
            };

            scope.register(Instruction::new(
                CoopMma::ExecuteScaled {
                    matrix,
                    registers_a,
                    registers_b,
                    registers_c,
                    scales_a: scales_a.expand,
                    scales_b: scales_b.expand,
                    scales_factor: self
                        .scales_factor
                        .expect("Can't execute scaled on matrix with no scales"),
                },
                registers_d.__extract_list(scope),
            ));

            registers_d
        })
    }
}

/// Fill the matrix with the provided value.
#[allow(unused_variables)]
pub fn fill<C: Scalar, S: MatrixScope>(mat: &mut Matrix<C, S>, value: C) {
    unexpanded!()
}

/// Module containing the expand function for [`fill()`].
pub mod fill {
    use super::*;

    /// Expand method of [`fill()`].
    pub fn expand<C: Scalar, S: MatrixScope>(
        scope: &Scope,
        mat: &mut MatrixExpand<C, S>,
        value: NativeExpand<C>,
    ) {
        let value: Variable = value.into();
        scope.register(Instruction::new(ir::CoopMma::Fill { value }, mat.elem));
    }
}

/// Load the matrix with the provided array using the stride.
#[allow(unused_variables)]
pub fn load<C: CubePrimitive, V: CubePrimitive, S: MatrixScope>(
    mat: &mut Matrix<C, S>,
    value: &[V],
    stride: u32,
) {
    unexpanded!()
}

/// Module containing the expand function for [`load()`].
pub mod load {
    use super::*;

    /// Expand method of [`load()`].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, V: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        mat: &mut MatrixExpand<C, S>,
        value: &SliceExpand<V>,
        stride: NativeExpand<u32>,
    ) {
        let stride: Variable = stride.into();
        assert_ne!(
            mat.ident,
            MatrixIdent::Accumulator,
            "Loading accumulator requires explicit layout. Use `load_with_layout` instead."
        );

        let ptr = unsafe { *value.__expand_as_ptr_method(scope) }.expand;

        scope.register(Instruction::new(
            ir::CoopMma::Load {
                ptr,
                stride,
                layout: None,
            },
            mat.elem,
        ));
    }
}

/// Load the matrix with the provided array using the tensor layout.
#[allow(unused_variables)]
pub fn load_tensor<C: CubePrimitive, V: CubePrimitive, S: MatrixScope>(
    mat: &mut Matrix<C, S>,
    value: &TensorView<V>,
) {
    unexpanded!()
}

/// Module containing the expand function for [`load_tensor()`].
pub mod load_tensor {
    use super::*;

    /// Expand method of [`load()`].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, V: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        mat: &mut MatrixExpand<C, S>,
        value: &TensorViewExpand<V>,
    ) {
        assert_ne!(
            mat.ident,
            MatrixIdent::Accumulator,
            "Loading accumulator requires explicit layout. Use `load_with_layout` instead."
        );
        let buffer = value.buffer.__extract_list(scope);

        scope.register(Instruction::new(
            ir::CoopMma::LoadTensor {
                buffer,
                layout: value.layout.expand,
                view: match &value.view {
                    ComptimeOptionExpand::None => None,
                    ComptimeOptionExpand::Some(view) => Some(view.expand),
                },
            },
            mat.elem,
        ));
    }
}

/// Load the matrix with the provided array using the stride with an explicit layout.
/// Explicit layouts are required when loading accumulators.
#[allow(unused_variables)]
pub fn load_with_layout<C: CubePrimitive, V: CubePrimitive, S: MatrixScope>(
    mat: &mut Matrix<C, S>,
    value: &[V],
    stride: u32,
    layout: MatrixLayout,
) {
    unexpanded!()
}

/// Module containing the expand function for [`load_with_layout()`].
pub mod load_with_layout {
    use super::*;

    /// Expand method of [`load_with_layout()`].
    #[allow(unused_variables)]
    pub fn expand<C: CubeType, V: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        mat: &mut MatrixExpand<C, S>,
        value: &SliceExpand<V>,
        stride: NativeExpand<u32>,
        layout: MatrixLayout,
    ) {
        let stride: Variable = stride.into();
        let ptr = unsafe { *value.__expand_as_ptr_method(scope) }.expand;

        scope.register(Instruction::new(
            ir::CoopMma::Load {
                ptr,
                stride,
                layout: Some(layout),
            },
            mat.elem,
        ));
    }
}

/// Store the matrix in the given array following the given stride and layout.
#[allow(unused_variables)]
pub fn store<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
    output: &mut [O],
    mat: &Matrix<C, S>,
    stride: u32,
    layout: MatrixLayout,
) {
    unexpanded!()
}

/// Module containing the expand function for [`store()`].
pub mod store {
    use super::*;

    /// Expand method of [`store()`].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        output: &mut SliceExpand<O>,
        mat: &MatrixExpand<C, S>,
        stride: NativeExpand<u32>,
        layout: MatrixLayout,
    ) {
        let stride: Variable = stride.into();

        let destination = unsafe { *output.__expand_as_ptr_method(scope) }.expand;

        scope.register(Instruction::no_out(ir::CoopMma::Store {
            mat: mat.elem,
            stride,
            destination,
            layout,
        }));
    }
}

/// Store the matrix in the given tensor view.
#[allow(unused_variables)]
pub fn store_tensor<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
    output: &mut TensorView<O>,
    mat: &Matrix<C, S>,
) {
    unexpanded!()
}

/// Module containing the expand function for [`store_tensor()`].
pub mod store_tensor {
    use super::*;

    /// Expand method of [`store()`].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        output: &mut TensorViewExpand<O>,
        mat: &MatrixExpand<C, S>,
    ) {
        let buffer = output.buffer.__extract_list(scope);
        scope.register(Instruction::new(
            ir::CoopMma::StoreTensor {
                mat: mat.elem,
                layout: output.layout.expand,
                view: match &output.view {
                    ComptimeOptionExpand::None => None,
                    ComptimeOptionExpand::Some(view) => Some(view.expand),
                },
            },
            buffer,
        ));
    }
}

/// Execute the matrix-multiply and accumulate operation on the given [matrices](Matrix).
#[allow(unused_variables)]
pub fn execute<
    A: CubePrimitive,
    B: CubePrimitive,
    C: CubePrimitive,
    D: CubePrimitive,
    S: MatrixScope,
>(
    mat_a: &Matrix<A, S>,
    mat_b: &Matrix<B, S>,
    mat_c: &Matrix<C, S>,
    mat_d: &Matrix<D, S>,
) {
    unexpanded!()
}

/// Module containing the expand function for [`execute()`].
pub mod execute {
    use super::*;

    /// Expand method of [`execute()`].
    pub fn expand<
        A: CubePrimitive,
        B: CubePrimitive,
        C: CubePrimitive,
        D: CubePrimitive,
        S: MatrixScope,
    >(
        scope: &Scope,
        mat_a: &MatrixExpand<A, S>,
        mat_b: &MatrixExpand<B, S>,
        mat_c: &MatrixExpand<C, S>,
        mat_d: &MatrixExpand<D, S>,
    ) {
        scope.register(Instruction::new(
            ir::CoopMma::Execute {
                mat_a: mat_a.elem,
                mat_b: mat_b.elem,
                mat_c: mat_c.elem,
            },
            mat_d.elem,
        ));
    }
}

/// Cast the matrix fragment to a different type
#[allow(unused_variables)]
pub fn cast<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
    input: &Matrix<C, S>,
) -> Matrix<O, S> {
    unexpanded!()
}

/// Module containing the expand function for [`cast()`].
pub mod cast {
    use super::*;

    /// Expand method of [`cast()`].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        input: &MatrixExpand<C, S>,
    ) -> MatrixExpand<O, S> {
        let ident = input.ident;

        if core::any::TypeId::of::<C>() == core::any::TypeId::of::<O>() {
            return MatrixExpand {
                elem: input.elem,
                ident,
                _c: PhantomData,
                _s: PhantomData,
            };
        }
        let input = input.elem;
        let input_mat = match input.kind {
            ir::VariableKind::Matrix { mat, .. } => mat,
            _ => unreachable!(),
        };

        let elem = O::__expand_as_type(scope).storage_type();
        let elem = scope.create_matrix(ir::Matrix::new(
            ident,
            input_mat.m,
            input_mat.n,
            input_mat.k,
            elem,
            MatrixLayout::Undefined,
            input_mat.scope,
        ));

        let output = MatrixExpand {
            ident,
            elem,
            _c: PhantomData,
            _s: PhantomData,
        };
        scope.register(Instruction::new(ir::CoopMma::Cast { input }, output.elem));

        output
    }
}

/// Cast the matrix fragment to a different type and a different matrix ident.
/// This allows casting to otherwise unsupported types, i.e. casting an f32 accumulator to bf16
/// (which can't be used as the accumulator type).
#[allow(unused_variables)]
pub fn cast_with_ident<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
    input: &Matrix<C, S>,
    ident: MatrixIdent,
) -> Matrix<O, S> {
    unexpanded!()
}

/// Module containing the expand function for [`cast()`].
pub mod cast_with_ident {
    use super::*;

    /// Expand method of [`cast()`].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        input: MatrixExpand<C, S>,
        ident: MatrixIdent,
    ) -> MatrixExpand<O, S> {
        if core::any::TypeId::of::<C>() == core::any::TypeId::of::<O>() && ident == input.ident {
            return MatrixExpand {
                elem: input.elem,
                ident,
                _c: PhantomData,
                _s: PhantomData,
            };
        }
        let input = input.elem;
        let input_mat = match input.kind {
            ir::VariableKind::Matrix { mat, .. } => mat,
            _ => unreachable!(),
        };

        let elem = O::__expand_as_type(scope).storage_type();
        let elem = scope.create_matrix(ir::Matrix::new(
            ident,
            input_mat.m,
            input_mat.n,
            input_mat.k,
            elem,
            MatrixLayout::Undefined,
            input_mat.scope,
        ));

        let output = MatrixExpand {
            ident,
            elem,
            _c: PhantomData,
            _s: PhantomData,
        };
        scope.register(Instruction::new(ir::CoopMma::Cast { input }, output.elem));

        output
    }
}

impl CubeType for MatrixLayout {
    type ExpandType = Self;
}

impl IntoExpand for MatrixLayout {
    type Expand = Self;

    fn into_expand(self, _scope: &Scope) -> Self::Expand {
        self
    }
}

impl ExpandTypeClone for MatrixLayout {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

impl IntoMut for MatrixLayout {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl CubeDebug for MatrixLayout {}

impl AsRefExpand for MatrixLayout {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}
impl AsMutExpand for MatrixLayout {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

/// Execute an elementwise op on the matrix fragment.
///
/// Function parameters are (row, col, element) -> element
#[allow(unused_variables)]
pub fn execute_elementwise_op<A: CubePrimitive, S: MatrixScope>(
    matrix_in: &Matrix<A, S>,
    matrix_out: &Matrix<A, S>,
    op: impl Fn(u32, u32, A::Scalar) -> A::Scalar,
) {
    unexpanded!()
}

/// Module containing the expand function for [`execute()`].
pub mod execute_elementwise_op {
    use alloc::vec;

    use super::*;

    /// Expand method of [`execute()`].
    pub fn expand<A: CubePrimitive, S: MatrixScope>(
        scope: &Scope,
        matrix_in: &MatrixExpand<A, S>,
        matrix_out: &MatrixExpand<A, S>,
        mut op: impl FnMut(
            &Scope,
            NativeExpand<u32>,
            NativeExpand<u32>,
            NativeExpand<A::Scalar>,
        ) -> NativeExpand<A::Scalar>,
    ) {
        let row = scope.create_local(u32::__expand_as_type(scope));
        let col = scope.create_local(u32::__expand_as_type(scope));
        let elem = scope.create_local(A::Scalar::__expand_as_type(scope));

        let mut closure_scope = scope.child();
        let return_value = op(&mut closure_scope, row.into(), col.into(), elem.into());
        closure_scope.return_value = Some(return_value.expand);

        let op = scope.create_function(vec![row, col, elem], closure_scope);

        scope.register(Instruction::new(
            ir::CoopMma::ExecuteElementwise {
                matrix: matrix_in.elem,
                op,
            },
            matrix_out.elem,
        ));
    }
}
