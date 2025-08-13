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
pub struct MmaDefinition<AB: CubeType, CD: CubeType> {
    _ab: PhantomData<AB>,
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
pub struct MmaDefinitionExpand<AB: CubeType, CD: CubeType> {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub ab_elem: Elem,
    pub cd_elem: Elem,
    _ab: PhantomData<AB>,
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

impl<AB: CubeType, CD: CubeType> Clone for MmaDefinitionExpand<AB, CD> {
    fn clone(&self) -> Self {
        Self {
            m: self.m,
            n: self.n,
            k: self.k,
            ab_elem: self.ab_elem,
            cd_elem: self.cd_elem,
            _ab: PhantomData,
            _cd: PhantomData,
        }
    }
}

impl<C: CubeType> CubeType for Matrix<C> {
    type ExpandType = MatrixExpand<C>;
}

impl<AB: CubeType, CD: CubeType> CubeType for MmaDefinition<AB, CD> {
    type ExpandType = MmaDefinitionExpand<AB, CD>;
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

impl<AB: CubeType, CD: CubeType> IntoMut for MmaDefinitionExpand<AB, CD> {
    fn into_mut(self, _scope: &mut Scope) -> Self {
        self
    }
}

impl<AB: CubeType, CD: CubeType> CubeDebug for MmaDefinitionExpand<AB, CD> {}

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
impl<AB: CubePrimitive, CD: CubePrimitive> MmaDefinition<AB, CD> {
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
            let ab_elem = AB::as_elem(scope);
            let cd_elem = CD::as_elem(scope);

            MmaDefinitionExpand {
                m,
                n,
                k,
                ab_elem,
                cd_elem,
                _ab: PhantomData,
                _cd: PhantomData,
            }
        })
    }

    /// Number of elements in the matrix
    #[allow(unused)]
    pub fn num_elems(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            match ident {
                MatrixIdent::A => self.m * self.k,
                MatrixIdent::B => self.k * self.n,
                MatrixIdent::Accumulator => self.m * self.n,
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
                MatrixIdent::A | MatrixIdent::B => Elem::size_bits(&self.ab_elem) as u32,
                MatrixIdent::Accumulator => Elem::size_bits(&self.cd_elem) as u32,
            };
            let register_size = scope.runtime_properties.mma.register_size_bits;
            // div_ceil for potential compatiblity with f64
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
                MatrixIdent::A | MatrixIdent::B => self.ab_elem,
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
                elem,
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

    /// Manually execute a low level `mma` operation with manually managed registers. Register layout
    /// and index mapping can be retrieved from the [`MatrixDefinition`]
    #[allow(unused)]
    pub fn execute(
        &self,
        a_registers: &Sequence<Line<AB>>,
        b_registers: &Sequence<Line<AB>>,
        c_registers: &Sequence<Line<CD>>,
    ) -> Array<Line<CD>> {
        intrinsic!(|scope| {
            let acc_elems = self
                .clone()
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let acc_line_size = self
                .clone()
                .__expand_line_size_method(scope, MatrixIdent::Accumulator);
            let num_registers = acc_elems / acc_line_size;

            let d_registers = Array::__expand_vectorized(scope, num_registers, acc_line_size);

            let a_registers = a_registers
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();
            let b_registers = b_registers
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();
            let c_registers = c_registers
                .iter_cloned()
                .map(|it| *it.expand)
                .collect::<Vec<_>>();

            // Only shape is actually used
            let matrix = cubecl_ir::Matrix {
                ident: MatrixIdent::A,
                m: self.m,
                n: self.n,
                k: self.k,
                elem: self.ab_elem,
                layout: MatrixLayout::ColMajor,
            };

            scope.register(Instruction::new(
                CoopMma::ExecuteManual {
                    matrix,
                    a_registers,
                    b_registers,
                    c_registers,
                },
                *d_registers.expand,
            ));

            d_registers
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
