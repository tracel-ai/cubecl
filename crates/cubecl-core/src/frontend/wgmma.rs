//! Warpgroup matrix-multiply and accumulate, Hopper's asynchronous tensor core instructions.
//!
//! The four planes of a warpgroup (128 contiguous units, starting at a multiple of 128) issue one
//! `64 x n x k` MMA together. `B`, and optionally `A`, are read straight from shared memory
//! through [matrix descriptors](descriptor), and the accumulator stays in registers, spread over
//! the warpgroup as [`WgmmaDefinition::position_of_nth`] describes.
//!
//! The MMA runs asynchronously. Its accumulator belongs to the tensor cores from the moment it is
//! issued until a [`wait_group`] retires it, so it must not be read or written in between. A
//! typical step looks like:
//!
//! ```rust, ignore
//! let def = WgmmaDefinition::<f16, f16, f32>::new(128usize);
//! let acc_len = def.elems_per_unit(MatrixIdent::Accumulator);
//! let size!(N) = 2usize;
//! let mut acc = Array::<Vector<f32, N>>::new(comptime![acc_len / 2]);
//!
//! // Shared memory written by the units must be made visible to the tensor cores.
//! sync_async_proxy_shared();
//! sync_cube();
//!
//! // Settle every write to the accumulator before the fence that orders it for the MMA.
//! wgmma::fence_operand(&mut acc);
//! wgmma::fence();
//! for k in 0..num_k_tiles {
//!     let a = wgmma::descriptor(&lhs_tile[k], lbo, sbo, Swizzle::B128);
//!     let b = wgmma::descriptor(&rhs_tile[k], lbo, sbo, Swizzle::B128);
//!     def.execute(a, b, &mut acc, k > 0, MatrixLayout::RowMajor, MatrixLayout::ColMajor);
//! }
//! wgmma::commit_group();
//! wgmma::wait_group(0usize);
//! wgmma::fence_operand(&mut acc);
//! ```
//!
//! Requires [`wgmma`](cubecl_ir::features::MatmulFeatures::wgmma), only present on `sm_90a`.

use crate::unexpanded;
use crate::{self as cubecl, prelude::*};
use core::marker::PhantomData;
use cubecl_ir::{
    Scope,
    dialect::matrix::{
        WgmmaCommitGroupOp, WgmmaDescriptorOp, WgmmaFenceOp, WgmmaFenceOperandOp, WgmmaOp,
        WgmmaWaitGroupOp,
    },
};
use cubecl_macros::{comptime_type, cube, intrinsic};
use pliron::builtin::attributes::TypeAttr;

use super::{CubeDebug, CubeType, IntoMut};
pub use cubecl_ir::dialect::matrix::WgmmaSwizzle as Swizzle;
pub use cubecl_ir::types::matrix::{MatrixIdent, MatrixLayout, MatrixShape};

use alloc::format;

/// Units in a warpgroup.
pub const WARPGROUP_SIZE: u32 = 128;
/// Rows every warpgroup MMA computes.
pub const WARPGROUP_M: usize = 64;
/// Bytes of K every warpgroup MMA reads per row.
const K_BYTES: usize = 32;

/// Builds the matrix descriptor of a shared memory tile, which a warpgroup MMA reads `A` or `B`
/// through. `tile` starts at the first element of the tile, and must be in shared memory and
/// aligned to [`Swizzle::alignment`]: 16 bytes without swizzle, and a full repeat of the
/// pattern with one, 1024 bytes for [`Swizzle::B128`]. A swizzled tile that starts anywhere else
/// is read with its rows permuted wrong, silently.
///
/// The offsets are in bytes and must be multiples of 16. Without swizzle the tile is made of
/// 8x16-byte core matrices: `leading_byte_offset` steps between core matrices along K and
/// `stride_byte_offset` between core matrices along M or N. With swizzle,
/// `stride_byte_offset` steps between groups of 8 rows, and `leading_byte_offset` is only read
/// for MN-major tiles.
#[cube]
#[allow(unused_variables)]
pub fn descriptor<E: CubePrimitive>(
    tile: &[E],
    leading_byte_offset: u32,
    stride_byte_offset: u32,
    #[comptime] swizzle: Swizzle,
) -> u64 {
    intrinsic!(|scope| {
        let ptr = unsafe { *tile.__expand_as_ptr_method(scope) }.value(scope);
        let leading_byte_offset = leading_byte_offset.read_value(scope);
        let stride_byte_offset = stride_byte_offset.read_value(scope);
        let op = WgmmaDescriptorOp::new(
            scope.ctx_mut(),
            ptr,
            leading_byte_offset,
            stride_byte_offset,
            swizzle,
        );
        scope.register_with_result(&op).into()
    })
}

/// Orders the warpgroup's register accesses before the MMAs that follow. Required before the
/// first MMA, and whenever an accumulator or `A` fragment an MMA reads was written by anything
/// else since the last fence.
pub fn fence() {
    unexpanded!()
}

/// Module containing the expand function for [`fence()`].
pub mod fence {
    use super::*;

    /// Expand method of [`fence()`].
    pub fn expand(scope: &Scope) {
        scope.register(&WgmmaFenceOp::new(scope.ctx_mut()))
    }
}

/// Commits every MMA issued since the last commit into one group.
pub fn commit_group() {
    unexpanded!()
}

/// Module containing the expand function for [`commit_group()`].
pub mod commit_group {
    use super::*;

    /// Expand method of [`commit_group()`].
    pub fn expand(scope: &Scope) {
        scope.register(&WgmmaCommitGroupOp::new(scope.ctx_mut()))
    }
}

/// Waits until at most `max_pending` committed groups are still running. The accumulators of
/// every older group may be read afterwards.
#[allow(unused_variables)]
pub fn wait_group(max_pending: usize) {
    unexpanded!()
}

/// Module containing the expand function for [`wait_group()`].
pub mod wait_group {
    use super::*;

    /// Expand method of [`wait_group()`].
    pub fn expand(scope: &Scope, max_pending: usize) {
        scope.register(&WgmmaWaitGroupOp::new(scope.ctx_mut(), max_pending))
    }
}

/// Pins `registers` in place: the compiler moves no access to them across this point.
///
/// PTX requires every write to an accumulator or `A` fragment to come before the [`fence`] that
/// precedes the MMA reading it, and every read to come after the [`wait_group`] that retires it.
/// The compiler sees neither rule, so pin the registers right before the [`fence`] and right
/// after the [`wait_group`]. Without it the MMAs stay correct, but `ptxas` may serialize them.
#[cube]
#[allow(unused_variables)]
pub fn fence_operand<E: Scalar, N: Size>(registers: &mut Array<Vector<E, N>>) {
    intrinsic!(|scope| {
        let registers = registers.__extract_list(scope);
        scope.register(&WgmmaFenceOperandOp::new(scope.ctx_mut(), registers))
    })
}

/// Defines a warpgroup MMA: its element types and its shape.
#[derive(Copy, Clone)]
pub struct WgmmaDefinition<A: CubeType, B: CubeType, CD: CubeType> {
    _a: PhantomData<A>,
    _b: PhantomData<B>,
    _cd: PhantomData<CD>,
}

/// Expand type of [`WgmmaDefinition`].
#[derive(Debug)]
pub struct WgmmaDefinitionExpand<A: CubeType, B: CubeType, CD: CubeType> {
    pub shape: MatrixShape,
    _a: PhantomData<A>,
    _b: PhantomData<B>,
    _cd: PhantomData<CD>,
}

impl<A: CubeType, B: CubeType, CD: CubeType> Copy for WgmmaDefinitionExpand<A, B, CD> {}
impl<A: CubeType, B: CubeType, CD: CubeType> Clone for WgmmaDefinitionExpand<A, B, CD> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> ExpandTypeClone for WgmmaDefinitionExpand<A, B, CD> {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> AsRefExpand for WgmmaDefinitionExpand<A, B, CD> {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> AsMutExpand for WgmmaDefinitionExpand<A, B, CD> {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> CubeType for WgmmaDefinition<A, B, CD> {
    type ExpandType = WgmmaDefinitionExpand<A, B, CD>;
}

impl<A: CubeType, B: CubeType, CD: CubeType> IntoExpand for WgmmaDefinitionExpand<A, B, CD> {
    type Expand = Self;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> IntoMut for WgmmaDefinitionExpand<A, B, CD> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<A: CubeType, B: CubeType, CD: CubeType> CubeDebug for WgmmaDefinitionExpand<A, B, CD> {}

#[cube]
impl<A: Scalar, B: Scalar, CD: Scalar> WgmmaDefinition<A, B, CD> {
    /// Defines a `64 x n x k` warpgroup MMA. Every warpgroup MMA computes 64 rows and reads 32
    /// bytes of K, so `k` follows `A`: 16 for `f16` and `bf16`, 8 for `tf32`, 32 for the 8-bit
    /// types. `n` is a multiple of 8 up to 256, or of 16 for the integer types.
    pub fn new(#[comptime] n: usize) -> Self {
        intrinsic!(|scope| {
            let (a, b, cd) = (
                A::elem_type(scope),
                B::elem_type(scope),
                CD::elem_type(scope),
            );
            let m = WARPGROUP_M;
            let k = K_BYTES / a.size();
            if let Some(props) = scope.state().device_properties.clone() {
                let wgmma = &props.features.matmul.wgmma;
                let supported = wgmma
                    .iter()
                    .any(|cfg| cfg.matches(a, b, cd, m as u32, n as u32, k as u32));
                if !supported {
                    scope.push_error(format!(
                        "the device doesn't support a {m}x{n}x{k} warpgroup MMA of {a:?} x \
                         {b:?} into {cd:?}; supported configurations: {wgmma:?}"
                    ));
                }
            }

            WgmmaDefinitionExpand {
                shape: (m, n, k).into(),
                _a: PhantomData,
                _b: PhantomData,
                _cd: PhantomData,
            }
        })
    }

    /// The `m x n x k` shape of the MMA.
    #[allow(unused)]
    pub fn shape(&self) -> comptime_type!(MatrixShape) {
        intrinsic!(|_| self.shape)
    }

    /// Elements each unit of the warpgroup holds: the `A` fragment for
    /// [`execute_registers_a`](Self::execute_registers_a), or the accumulator. `B` is always
    /// read from shared memory.
    #[allow(unused)]
    pub fn elems_per_unit(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|_| {
            let MatrixShape { m, n, k } = self.shape;
            match ident {
                MatrixIdent::A => m * k / WARPGROUP_SIZE as usize,
                MatrixIdent::Accumulator => m * n / WARPGROUP_SIZE as usize,
                MatrixIdent::B => panic!("a warpgroup MMA always reads B from shared memory"),
            }
        })
    }

    /// Elements of a unit that are contiguous in a row: 32 bits of `A`, or two accumulator
    /// elements. A unit's elements go in runs of this length.
    #[allow(unused)]
    pub fn contiguous_elems(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(u32) {
        intrinsic!(|scope| {
            match ident {
                MatrixIdent::A => 4 / A::elem_type(scope).size() as u32,
                MatrixIdent::Accumulator => 2,
                MatrixIdent::B => panic!("a warpgroup MMA always reads B from shared memory"),
            }
        })
    }

    /// The `(row, col)` of the `nth` element `unit` holds of the `A` fragment or the
    /// accumulator. `unit` is the unit's position in its warpgroup, `0..128`.
    ///
    /// Each plane holds 16 rows. A unit holds runs of [`Self::contiguous_elems`] elements in its
    /// row `lane / 4` of them, then the same in row `lane / 4 + 8`, then moves on 8 columns of
    /// 32-bit words.
    pub fn position_of_nth(
        &self,
        unit: u32,
        nth: u32,
        #[comptime] ident: MatrixIdent,
    ) -> (u32, u32) {
        let run = self.contiguous_elems(ident);
        let plane = unit / 32;
        let lane = unit % 32;
        let chunk = nth / run;
        let row = plane * 16 + lane / 4 + (chunk % 2) * 8;
        let col = (chunk / 2) * (run * 4) + (lane % 4) * run + nth % run;
        (row, col)
    }

    /// Issues `D = A * B + D` with both operands in shared memory, or `D = A * B` when
    /// `scale_d` is false. `a` and `b` are [matrix descriptors](descriptor).
    ///
    /// A row-major `A` and a column-major `B` are K-major, the layout every element type takes.
    /// The other layouts are for `f16` and `bf16` alone.
    ///
    /// The MMA is asynchronous: `acc` must not be touched until a [`wait_group`] retires it.
    #[allow(unused_variables)]
    pub fn execute<N: Size>(
        &self,
        a: u64,
        b: u64,
        acc: &mut Array<Vector<CD, N>>,
        scale_d: bool,
        #[comptime] a_layout: MatrixLayout,
        #[comptime] b_layout: MatrixLayout,
    ) {
        intrinsic!(|scope| {
            let a = a.read_value(scope);
            self.__expand_issue_method(scope, a, b, acc, scale_d, a_layout, b_layout)
        })
    }

    /// Issues `D = A * B + D` with `A` in registers, laid out as [`Self::position_of_nth`]
    /// describes, and `B` in shared memory, or `D = A * B` when `scale_d` is false.
    ///
    /// The MMA is asynchronous: neither `a` nor `acc` may be touched until a [`wait_group`]
    /// retires it.
    #[allow(unused_variables)]
    pub fn execute_registers_a<NA: Size, N: Size>(
        &self,
        a: &Array<Vector<A, NA>>,
        b: u64,
        acc: &mut Array<Vector<CD, N>>,
        scale_d: bool,
        #[comptime] b_layout: MatrixLayout,
    ) {
        intrinsic!(|scope| {
            let a = a.read_value(scope);
            self.__expand_issue_method(scope, a, b, acc, scale_d, MatrixLayout::RowMajor, b_layout)
        })
    }
}

impl<A: Scalar, B: Scalar, CD: Scalar> WgmmaDefinitionExpand<A, B, CD> {
    #[allow(clippy::too_many_arguments)]
    fn __expand_issue_method<N: Size>(
        &self,
        scope: &Scope,
        a: cubecl_ir::pliron::value::Value,
        b: NativeExpand<u64>,
        acc: &mut NativeExpand<Array<Vector<CD, N>>>,
        scale_d: NativeExpand<bool>,
        a_layout: MatrixLayout,
        b_layout: MatrixLayout,
    ) {
        let b = b.read_value(scope);
        let scale_d = scale_d.read_value(scope);
        let acc = acc.__extract_list(scope);
        let a_ty = TypeAttr::new(A::__expand_as_type(scope));
        let b_ty = TypeAttr::new(B::__expand_as_type(scope));
        scope.register(&WgmmaOp::new(
            scope.ctx_mut(),
            a,
            b,
            acc,
            scale_d,
            a_ty,
            b_ty,
            self.shape,
            a_layout,
            b_layout,
        ));
    }
}
