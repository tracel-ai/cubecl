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
//! pub fn example(lhs: &Array<F16>, rhs: &Array<F16>, out: &mut Array<F32>) {
//!     let a = cmma::Matrix::<F16>::new(
//!         cmma::MatrixIdent::A,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::RowMajor,
//!     );
//!     let b = cmma::Matrix::<F16>::new(
//!         cmma::MatrixIdent::B,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::ColMajor,
//!     );
//!     let c = cmma::Matrix::<F32>::new(
//!         cmma::MatrixIdent::Accumulator,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::Undefined,
//!     );
//!     cmma::fill::<F32>(&c, F32::new(0.0));
//!     cmma::load::<F16>(&a, lhs.as_slice(), u32::new(16));
//!     cmma::load::<F16>(&b, rhs.as_slice(), u32::new(16));
//!
//!     cmma::execute::<F16, F16, F32, F32>(&a, &b, &c, &c);
//!
//!     cmma::store::<F32>(
//!         out.as_slice_mut(),
//!         &c,
//!         u32::new(16),
//!         cmma::MatrixLayout::RowMajor,
//!     );
//! }
//! ```

use super::{
    CubeDebug, CubePrimitive, CubeType, ExpandElementTyped, IntoMut, ReadOnly, Slice, SliceExpand,
    SliceMut,
};
use crate::{
    self as cubecl,
    prelude::{Array, Line, ReadWrite},
};
use crate::{
    ir::{self, Instruction},
    unexpanded,
};
use cubecl_macros::{comptime_type, cube, intrinsic};
use std::marker::PhantomData;

use cubecl_ir::{CoopMma, ExpandElement, LineSize, Scope, StorageType, Type};
pub use ir::{MatrixIdent, MatrixLayout};

/// A matrix represent a 2D grid of numbers.
///
/// They can either be in a [row major](MatrixLayout::RowMajor) or a
/// [column major](MatrixLayout::ColMajor) format.
#[derive(Copy, Clone)]
pub struct Matrix<C: CubeType> {
    _c: PhantomData<C>,
}

/// Defines a matrix multiplication operation, including the input and output type, and the shape.
#[derive(Copy, Clone)]
pub struct MmaDefinition<A: CubeType, B: CubeType, CD: CubeType> {
    _a: PhantomData<A>,
    _b: PhantomData<B>,
    _cd: PhantomData<CD>,
}

/// Expand type of [Matrix].
pub struct MatrixExpand<C: CubeType> {
    elem: ExpandElement,
    ident: MatrixIdent,
    _c: PhantomData<C>,
}

/// Expand type of [MmaDefinition].
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

impl<C: CubeType> Clone for MatrixExpand<C> {
    fn clone(&self) -> Self {
        Self {
            elem: self.elem.clone(),
            ident: self.ident,
            _c: self._c,
        }
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> Clone for MmaDefinitionExpand<A, B, CD> {
    fn clone(&self) -> Self {
        Self {
            m: self.m,
            n: self.n,
            k: self.k,
            a_type: self.a_type,
            b_type: self.b_type,
            cd_type: self.cd_type,
            scales_factor: self.scales_factor,
            scales_type: self.scales_type,
            _a: PhantomData,
            _b: PhantomData,
            _cd: PhantomData,
        }
    }
}

impl<C: CubeType> CubeType for Matrix<C> {
    type ExpandType = MatrixExpand<C>;
}

impl<A: CubeType, B: CubeType, CD: CubeType> CubeType for MmaDefinition<A, B, CD> {
    type ExpandType = MmaDefinitionExpand<A, B, CD>;
}

impl<C: CubeType> IntoMut for MatrixExpand<C> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<C: CubeType> CubeDebug for MatrixExpand<C> {
    fn set_debug_name(&self, scope: &mut Scope, name: &'static str) {
        scope.update_variable_name(*self.elem, name);
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> IntoMut for MmaDefinitionExpand<A, B, CD> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> CubeDebug for MmaDefinitionExpand<A, B, CD> {}

#[cube]
impl<C: CubePrimitive> Matrix<C> {
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
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
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
            let elem = C::as_type(scope);
            let elem = scope.create_matrix(ir::Matrix::new(ident, m, n, k, elem, layout));
            MatrixExpand {
                elem,
                ident,
                _c: PhantomData,
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
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
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
    ) -> Self {
        let mat = unsafe { Self::uninitialized(ident, m, n, k, layout) };

        intrinsic!(|scope| {
            fill::expand(scope, mat.clone(), value);
            mat
        })
    }

    /// Create a new matrix that is going to be used in the
    /// [matrix-multiply and accumulate](execute()) function and is loaded from `value` with `stride`.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
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
        value: &Slice<C>,
        stride: u32,
    ) -> Self {
        let mat = unsafe { Self::uninitialized(ident, m, n, k, layout) };

        intrinsic!(|scope| {
            load::expand(scope, mat.clone(), value, stride);
            mat
        })
    }
}

#[cube]
impl<A: CubePrimitive, B: CubePrimitive, CD: CubePrimitive> MmaDefinition<A, B, CD> {
    /// Create a new matrix definition that is going to be used in the manual
    /// [matrix-multiply and accumulate](execute_manual()) function.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    /// Layout for manual MMA is determined by the runtime and must be handled manually.
    /// Use [`line_layout`] to check the correct data layout for each element.
    ///
    /// Refer to [nvidia documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#element-types-and-matrix-sizes).
    #[allow(unused_variables)]
    pub fn new(#[comptime] m: usize, #[comptime] n: usize, #[comptime] k: usize) -> Self {
        intrinsic!(|scope| {
            let a_type = A::as_type(scope);
            let b_type = B::as_type(scope);
            let cd_type = CD::as_type(scope);

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
    /// [matrix-multiply and accumulate](execute_manual()) function.
    ///
    /// You have to declare the shape used for the execution.
    /// The shape of the current matrix is determined using the [MatrixIdent].
    ///
    /// * [MatrixIdent::A] Shape => (M, K)
    /// * [MatrixIdent::B] Shape => (K, N)
    /// * [MatrixIdent::Accumulator] Shape => (M, N)
    ///
    /// Not all shapes are supported, and the permitted shapes depend on the element type.
    /// Layout for manual MMA is determined by the runtime and must be handled manually.
    /// Use [`line_layout`] to check the correct data layout for each element.
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
            let a_type = A::as_type(scope);
            let b_type = B::as_type(scope);
            let cd_type = CD::as_type(scope);

            MmaDefinitionExpand {
                m,
                n,
                k,
                a_type,
                b_type,
                cd_type,
                scales_factor: Some(scale_factor),
                scales_type: Some(S::as_type(scope)),
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

    /// Returns the number of elements handled by each lane. Should be packed into `Line`s of size
    /// `line_size` with [`line_layout`].
    ///
    /// # Note
    /// "Lane" here refers to the unit relative to a plane, to distinguish it from a unit relative
    /// to a cube.
    #[allow(unused)]
    pub fn elems_per_lane(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            let elems = self.__expand_num_elems_method(scope, ident);
            let plane_dim = scope.runtime_properties.mma.const_plane_size as usize;
            let duplication = match ident {
                MatrixIdent::A => scope.runtime_properties.mma.register_duplication_a,
                MatrixIdent::B => scope.runtime_properties.mma.register_duplication_b,
                MatrixIdent::Accumulator => scope.runtime_properties.mma.register_duplication_acc,
            };
            (elems / plane_dim) * duplication
        })
    }

    /// Returns the number of lines of size `line_size` with layout `line_layout` per lane.
    ///
    /// # Note
    /// "Lane" here refers to the unit relative to a plane, to distinguish it from a unit relative
    /// to a cube.
    #[allow(unused)]
    pub fn lines_per_lane(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            let elems = self.clone().__expand_elems_per_lane_method(scope, ident);
            let line_size = self.__expand_line_size_method(scope, ident);
            elems / line_size
        })
    }

    /// The layout of each line in this matrix (row major or column major)
    #[allow(unused)]
    pub fn line_layout(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(MatrixLayout) {
        intrinsic!(|scope| {
            match ident {
                MatrixIdent::A => scope.runtime_properties.mma.register_layout_a,
                MatrixIdent::B => scope.runtime_properties.mma.register_layout_b,
                MatrixIdent::Accumulator => scope.runtime_properties.mma.register_layout_acc,
            }
        })
    }

    /// Number of elements in each line passed to the execute function. Represents the maximum
    /// number of contiguous elements held by the thread.
    #[allow(unused_variables)]
    pub fn line_size(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(LineSize) {
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
            };
            scope
                .runtime_properties
                .mma
                .contiguous_elements
                .apply(ident, matrix)
        })
    }

    /// Returns the coordinates of the `nth` element handled by the `lane_id`
    /// Each lane contains [`elems_per_lane`] elements in [`line_size`] chunks.
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
            let lane_id: ExpandElement = lane_id.into();
            let elem_idx: ExpandElement = elem_idx.into();

            let ty = match ident {
                MatrixIdent::A => self.a_type,
                MatrixIdent::B => self.b_type,
                MatrixIdent::Accumulator => self.cd_type,
            };
            let layout = match ident {
                MatrixIdent::A => scope.runtime_properties.mma.register_layout_a,
                MatrixIdent::B => scope.runtime_properties.mma.register_layout_b,
                MatrixIdent::Accumulator => scope.runtime_properties.mma.register_layout_acc,
            };
            let matrix = cubecl_ir::Matrix {
                ident,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: ty,
                layout,
            };

            let row = scope.create_local(Type::new(u32::as_type(scope)));
            let col = scope.create_local(Type::new(u32::as_type(scope)));
            scope.register(Instruction::new(
                CoopMma::RowIndex {
                    lane_id: *lane_id,
                    i: *elem_idx,
                    matrix,
                },
                *row,
            ));
            scope.register(Instruction::new(
                CoopMma::ColIndex {
                    lane_id: *lane_id,
                    i: *elem_idx,
                    matrix,
                },
                *col,
            ));
            (row.into(), col.into())
        })
    }

    /// Index of the scales for this thread, along the non-major dimension of the matrix.
    /// Each thread loads all scales in the major direction into a single `Line`.
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

    /// Number of scales in each line (not the line size!). Line size may include padding bytes.
    pub fn scales_count(&self) -> comptime_type!(usize) {
        // We only have the CUDA version for now, so just use `scales_factor`. The function can
        // be modified for HIP in the future without having to redo all uses.
        intrinsic!(|_| {
            self.scales_factor
                .expect("Can't retrieve scales count for matrix with no scales")
        })
    }

    /// Line size for the scale factors. May be larger than the total number of scales.
    pub fn scales_line_size(&self) -> comptime_type!(LineSize) {
        intrinsic!(|scope| {
            let elem = self
                .scales_type
                .expect("Can't retrieve scales line size for matrix with no scales");
            scope.runtime_properties.mma.register_size_bits / elem.size_bits()
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
    pub fn load_matrix<E: CubePrimitive>(
        &self,
        row: &Slice<Line<E>>,
        #[comptime] ident: MatrixIdent,
        #[comptime] num_matrices: usize,
        #[comptime] transpose: bool,
    ) -> Array<Line<E>> {
        intrinsic!(|scope| {
            let line_size = self.__expand_line_size_method(scope, ident);
            let slice_line_size = row.line_size;
            let (buffer, offset) = row.__to_raw_parts();
            let out = Array::__expand_lined(scope, num_matrices, line_size);
            scope.register(Instruction::new(
                CoopMma::LoadMatrix {
                    buffer,
                    offset,
                    line_size: slice_line_size,
                    factor: num_matrices,
                    transpose,
                },
                *out.expand,
            ));
            out
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
    pub fn store_matrix<E: CubePrimitive>(
        &self,
        row: &mut Slice<Line<E>, ReadWrite>,
        registers: &Array<Line<E>>,
        #[comptime] ident: MatrixIdent,
        #[comptime] num_matrices: usize,
        #[comptime] transpose: bool,
    ) {
        intrinsic!(|scope| {
            let line_size = self.__expand_line_size_method(scope, ident);
            let slice_line_size = row.line_size;
            let (buffer, offset) = row.__to_raw_parts();
            scope.register(Instruction::new(
                CoopMma::StoreMatrix {
                    offset,
                    line_size: slice_line_size,
                    registers: *registers.expand,
                    factor: num_matrices,
                    transpose,
                },
                buffer,
            ));
        })
    }

    /// Execute a low level `mma` operation with manually managed registers. Register layout
    /// and index mapping can be retrieved from the [`MatrixDefinition`]
    #[allow(unused)]
    pub fn execute(
        &self,
        registers_a: &Array<Line<A>>,
        registers_b: &Array<Line<B>>,
        registers_c: &Array<Line<CD>>,
    ) -> Array<Line<CD>> {
        intrinsic!(|scope| {
            let acc_elems = self
                .clone()
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let acc_line_size = self
                .clone()
                .__expand_line_size_method(scope, MatrixIdent::Accumulator);
            let num_registers = acc_elems / acc_line_size;

            let registers_d = Array::__expand_lined(scope, num_registers, acc_line_size);

            let registers_a = *registers_a.expand;
            let registers_b = *registers_b.expand;
            let registers_c = *registers_c.expand;

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: self.a_type,
                layout: MatrixLayout::ColMajor,
            };

            scope.register(Instruction::new(
                CoopMma::ExecuteManual {
                    matrix,
                    registers_a,
                    registers_b,
                    registers_c,
                },
                *registers_d.expand,
            ));

            registers_d
        })
    }

    /// Execute a low level block scaled `mma` operation with manually managed registers. Register
    /// layout and index mapping can be retrieved from the [`MatrixDefinition`]
    #[allow(unused)]
    pub fn execute_scaled<S: CubePrimitive>(
        &self,
        registers_a: &Array<Line<A>>,
        registers_b: &Array<Line<B>>,
        registers_c: &Array<Line<CD>>,
        scales_a: Line<S>,
        scales_b: Line<S>,
    ) -> Array<Line<CD>> {
        intrinsic!(|scope| {
            let acc_elems = self
                .clone()
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let acc_line_size = self
                .clone()
                .__expand_line_size_method(scope, MatrixIdent::Accumulator);
            let num_registers = acc_elems / acc_line_size;

            let registers_d = Array::__expand_lined(scope, num_registers, acc_line_size);

            let registers_a = *registers_a.expand;
            let registers_b = *registers_b.expand;
            let registers_c = *registers_c.expand;

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                storage: self.a_type,
                layout: MatrixLayout::ColMajor,
            };

            scope.register(Instruction::new(
                CoopMma::ExecuteScaled {
                    matrix,
                    registers_a,
                    registers_b,
                    registers_c,
                    scales_a: *scales_a.expand,
                    scales_b: *scales_b.expand,
                    scales_factor: self
                        .scales_factor
                        .expect("Can't execute scaled on matrix with no scales"),
                },
                *registers_d.expand,
            ));

            registers_d
        })
    }
}

/// Fill the matrix with the provided value.
#[allow(unused_variables)]
pub fn fill<C: CubeType>(mat: &Matrix<C>, value: C) {
    unexpanded!()
}

/// Module containing the expand function for [fill()].
pub mod fill {
    use super::*;

    /// Expand method of [fill()].
    pub fn expand<C: CubeType>(
        scope: &mut Scope,
        mat: MatrixExpand<C>,
        value: ExpandElementTyped<C>,
    ) {
        let value: ExpandElement = value.into();
        scope.register(Instruction::new(
            ir::CoopMma::Fill { value: *value },
            *mat.elem,
        ));
    }
}

/// Load the matrix with the provided array using the stride.
#[allow(unused_variables)]
pub fn load<C: CubePrimitive, V: CubePrimitive>(mat: &Matrix<C>, value: &Slice<V>, stride: u32) {
    unexpanded!()
}

/// Module containing the expand function for [load()].
pub mod load {
    use super::*;

    /// Expand method of [load()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, V: CubePrimitive>(
        scope: &mut Scope,
        mat: MatrixExpand<C>,
        value: SliceExpand<V, ReadOnly>,
        stride: ExpandElementTyped<u32>,
    ) {
        let stride: ExpandElement = stride.into();
        assert_ne!(
            mat.ident,
            MatrixIdent::Accumulator,
            "Loading accumulator requires explicit layout. Use `load_with_layout` instead."
        );

        let (value, offset) = value.__to_raw_parts();

        scope.register(Instruction::new(
            ir::CoopMma::Load {
                value,
                stride: *stride,
                offset,
                layout: None,
            },
            *mat.elem,
        ));
    }
}

/// Load the matrix with the provided array using the stride with an explicit layout.
/// Explicit layouts are required when loading accumulators.
#[allow(unused_variables)]
pub fn load_with_layout<C: CubePrimitive, V: CubePrimitive>(
    mat: &Matrix<C>,
    value: &Slice<V>,
    stride: u32,
    layout: MatrixLayout,
) {
    unexpanded!()
}

/// Module containing the expand function for [load_with_layout()].
pub mod load_with_layout {
    use super::*;

    /// Expand method of [load_with_layout()].
    #[allow(unused_variables)]
    pub fn expand<C: CubeType, V: CubePrimitive>(
        scope: &mut Scope,
        mat: MatrixExpand<C>,
        value: SliceExpand<V, ReadOnly>,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();
        let (value, offset) = value.__to_raw_parts();

        scope.register(Instruction::new(
            ir::CoopMma::Load {
                value,
                stride: *stride,
                offset,
                layout: Some(layout),
            },
            *mat.elem,
        ));
    }
}

/// Store the matrix in the given array following the given stride and layout.
#[allow(unused_variables)]
pub fn store<C: CubePrimitive, O: CubePrimitive>(
    output: &mut SliceMut<O>,
    mat: &Matrix<C>,
    stride: u32,
    layout: MatrixLayout,
) {
    unexpanded!()
}

/// Module containing the expand function for [store()].
pub mod store {
    use crate::prelude::ReadWrite;

    use super::*;

    /// Expand method of [store()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive>(
        scope: &mut Scope,
        output: SliceExpand<O, ReadWrite>,
        mat: MatrixExpand<C>,
        stride: ExpandElementTyped<u32>,
        layout: MatrixLayout,
    ) {
        let stride: ExpandElement = stride.into();

        let (output, offset) = output.__to_raw_parts();

        scope.register(Instruction::new(
            ir::CoopMma::Store {
                mat: *mat.elem,
                offset,
                stride: *stride,
                layout,
            },
            output,
        ));
    }
}

/// Execute the matrix-multiply and accumulate operation on the given [matrices](Matrix).
#[allow(unused_variables)]
pub fn execute<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
    mat_a: &Matrix<A>,
    mat_b: &Matrix<B>,
    mat_c: &Matrix<C>,
    mat_d: &Matrix<D>,
) {
    unexpanded!()
}

/// Module containing the expand function for [execute()].
pub mod execute {
    use super::*;

    /// Expand method of [execute()].
    pub fn expand<A: CubePrimitive, B: CubePrimitive, C: CubePrimitive, D: CubePrimitive>(
        scope: &mut Scope,
        mat_a: MatrixExpand<A>,
        mat_b: MatrixExpand<B>,
        mat_c: MatrixExpand<C>,
        mat_d: MatrixExpand<D>,
    ) {
        scope.register(Instruction::new(
            ir::CoopMma::Execute {
                mat_a: *mat_a.elem,
                mat_b: *mat_b.elem,
                mat_c: *mat_c.elem,
            },
            *mat_d.elem,
        ));
    }
}

/// Store the matrix in the given array following the given stride and layout.
#[allow(unused_variables)]
pub fn cast<C: CubePrimitive, O: CubePrimitive>(input: &Matrix<C>) -> Matrix<O> {
    unexpanded!()
}

/// Module containing the expand function for [store()].
pub mod cast {
    use super::*;

    /// Expand method of [store()].
    #[allow(unused_variables)]
    pub fn expand<C: CubePrimitive, O: CubePrimitive>(
        scope: &mut Scope,
        input: MatrixExpand<C>,
    ) -> MatrixExpand<O> {
        let ident = input.ident;

        if core::any::TypeId::of::<C>() == core::any::TypeId::of::<O>() {
            return MatrixExpand {
                elem: input.elem,
                ident,
                _c: PhantomData,
            };
        }
        let input = *input.elem;
        let input_mat = match input.kind {
            ir::VariableKind::Matrix { mat, .. } => mat,
            _ => unreachable!(),
        };

        let elem = O::as_type(scope);
        let elem = scope.create_matrix(ir::Matrix::new(
            ident,
            input_mat.m,
            input_mat.n,
            input_mat.k,
            elem,
            MatrixLayout::Undefined,
        ));

        let output = MatrixExpand {
            ident,
            elem,
            _c: PhantomData,
        };
        scope.register(Instruction::new(ir::CoopMma::Cast { input }, *output.elem));

        output
    }
}

impl CubeType for MatrixLayout {
    type ExpandType = Self;
}

impl IntoMut for MatrixLayout {
    fn into_mut(self, _scope: &mut crate::ir::Scope) -> Self {
        self
    }
}

impl CubeDebug for MatrixLayout {}
