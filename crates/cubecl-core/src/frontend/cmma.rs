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
    prelude::{Array, Line, Sequence},
};
use crate::{
    ir::{self, Instruction},
    unexpanded,
};
use cubecl_macros::{comptime_type, cube, intrinsic};
use std::marker::PhantomData;

use cubecl_ir::{CoopMma, Elem, ExpandElement, Item, Scope};
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
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_elem: Elem,
    pub b_elem: Elem,
    pub cd_elem: Elem,
    pub scales_factor: Option<u32>,
    pub scales_elem: Option<Elem>,
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
            a_elem: self.a_elem,
            b_elem: self.b_elem,
            cd_elem: self.cd_elem,
            scales_factor: self.scales_factor,
            scales_elem: self.scales_elem,
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
        m: u32,
        n: u32,
        k: u32,
        layout: MatrixLayout,
    ) -> Self {
        intrinsic!(|scope| {
            let elem = C::as_elem(scope);
            let elem = scope.create_matrix(ir::Matrix::new(
                ident,
                m.constant().unwrap().as_u32(),
                n.constant().unwrap().as_u32(),
                k.constant().unwrap().as_u32(),
                elem,
                layout,
            ));
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
        m: u32,
        n: u32,
        k: u32,
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
        m: u32,
        n: u32,
        k: u32,
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
    pub fn new(#[comptime] m: u32, #[comptime] n: u32, #[comptime] k: u32) -> Self {
        intrinsic!(|scope| {
            let a_elem = A::as_elem(scope);
            let b_elem = B::as_elem(scope);
            let cd_elem = CD::as_elem(scope);

            MmaDefinitionExpand {
                m,
                n,
                k,
                a_elem,
                b_elem,
                cd_elem,
                scales_factor: None,
                scales_elem: None,
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
        #[comptime] m: u32,
        #[comptime] n: u32,
        #[comptime] k: u32,
        #[comptime] scale_factor: u32,
    ) -> Self {
        intrinsic!(|scope| {
            let a_elem = A::as_elem(scope);
            let b_elem = B::as_elem(scope);
            let cd_elem = CD::as_elem(scope);

            MmaDefinitionExpand {
                m,
                n,
                k,
                a_elem,
                b_elem,
                cd_elem,
                scales_factor: Some(scale_factor),
                scales_elem: Some(S::as_elem(scope)),
                _a: PhantomData,
                _b: PhantomData,
                _cd: PhantomData,
            }
        })
    }

    /// Number of elements in the matrix
    #[allow(unused)]
    pub fn num_elems(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            match ident {
                MatrixIdent::A => (self.m * self.k) / self.a_elem.packing_factor() as u32,
                MatrixIdent::B => (self.k * self.n) / self.b_elem.packing_factor() as u32,
                MatrixIdent::Accumulator => {
                    (self.m * self.n) / self.cd_elem.packing_factor() as u32
                }
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
    pub fn elems_per_lane(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            let elems = self.__expand_num_elems_method(scope, ident);
            let plane_dim = scope.runtime_properties.mma.const_plane_size;
            let duplication = match ident {
                MatrixIdent::A => scope.runtime_properties.mma.register_duplication_a,
                MatrixIdent::B => scope.runtime_properties.mma.register_duplication_b,
                MatrixIdent::Accumulator => scope.runtime_properties.mma.register_duplication_acc,
            };
            (elems / plane_dim) * duplication
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

    /// Number of elements in each line passed to the execute function
    #[allow(unused_variables)]
    pub fn line_size(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            let bits = match ident {
                MatrixIdent::A => Elem::size_bits(&self.a_elem) as u32,
                MatrixIdent::B => Elem::size_bits(&self.b_elem) as u32,
                MatrixIdent::Accumulator => Elem::size_bits(&self.cd_elem) as u32,
            };
            let register_size = scope.runtime_properties.mma.register_size_bits;
            // div_ceil for potential compatibility with f64
            register_size.div_ceil(bits)
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
    pub fn indices_of_nth(
        &self,
        lane_id: u32,
        elem_idx: u32,
        #[comptime] ident: MatrixIdent,
    ) -> (u32, u32) {
        intrinsic!(|scope| {
            let lane_id: ExpandElement = lane_id.into();
            let elem_idx: ExpandElement = elem_idx.into();

            let elem = match ident {
                MatrixIdent::A => self.a_elem,
                MatrixIdent::B => self.b_elem,
                MatrixIdent::Accumulator => self.cd_elem,
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
                elem: elem.unpacked(),
                layout,
            };

            let row = scope.create_local(Item::new(u32::as_elem(scope)));
            let col = scope.create_local(Item::new(u32::as_elem(scope)));
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
    pub fn scales_count(&self) -> comptime_type!(u32) {
        // We only have the CUDA version for now, so just use `scales_factor`. The function can
        // be modified for HIP in the future without having to redo all uses.
        intrinsic!(|_| {
            self.scales_factor
                .expect("Can't retrieve scales count for matrix with no scales")
        })
    }

    /// Line size for the scale factors. May be larger than the total number of scales.
    pub fn scales_line_size(&self) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            let elem = self
                .scales_elem
                .expect("Can't retrieve scales line size for matrix with no scales");
            scope.runtime_properties.mma.register_size_bits / elem.size_bits() as u32
        })
    }

    /// Execute a low level `mma` operation with manually managed registers. Register layout
    /// and index mapping can be retrieved from the [`MatrixDefinition`]
    #[allow(unused)]
    pub fn execute(
        &self,
        registers_a: &Sequence<Line<A>>,
        registers_b: &Sequence<Line<B>>,
        registers_c: &Sequence<Line<CD>>,
    ) -> Array<Line<CD>> {
        intrinsic!(|scope| {
            let acc_elems = self
                .clone()
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let acc_line_size = self
                .clone()
                .__expand_line_size_method(scope, MatrixIdent::Accumulator);
            let num_registers = acc_elems / acc_line_size;

            let registers_d = Array::__expand_vectorized(scope, num_registers, acc_line_size);

            let registers_a = registers_a
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();
            let registers_b = registers_b
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();
            let registers_c = registers_c
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                elem: self.a_elem,
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
        registers_a: &Sequence<Line<A>>,
        registers_b: &Sequence<Line<B>>,
        registers_c: &Sequence<Line<CD>>,
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

            let registers_d = Array::__expand_vectorized(scope, num_registers, acc_line_size);

            let registers_a = registers_a
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();
            let registers_b = registers_b
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();
            let registers_c = registers_c
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                elem: self.a_elem,
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

        let elem = O::as_elem(scope);
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
