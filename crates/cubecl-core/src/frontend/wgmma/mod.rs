//! Warpgroup matrix-multiply and accumulate, Hopper's asynchronous tensor core instructions.
//!
//! The four planes of a warpgroup (128 contiguous units, starting at a multiple of 128) issue one
//! `64 x n x k` MMA together. `B`, and optionally `A`, are read straight from shared memory
//! through [matrix descriptors](MatrixDescriptor), and the [`Accumulator`] stays in registers,
//! spread over the warpgroup as [`Accumulator::position_of_nth`] describes.
//!
//! The MMAs run asynchronously, and own the accumulator while they do: [`Accumulator::start`]
//! hands it over as a [`Pending`], which issues the MMAs, and gives it back once they complete.
//! A K loop over shared memory stages looks like:
//!
//! ```rust, ignore
//! let layout = WgmmaTileLayout { major: Major::K, swizzle: Swizzle::B128, rows: 64, k: 64 };
//! let mut acc = Accumulator::<f32>::new(128usize).start();
//! for stage in 0..num_stages {
//!     // ... wait for the stage's tiles to land in shared memory ...
//!     let a = MatrixDescriptor::new(&lhs_tile[stage], layout);
//!     let b = MatrixDescriptor::new(&rhs_tile[stage], layout_b);
//!     #[unroll]
//!     for k in 0..steps {
//!         acc.execute(&a.at(0, k * k_step), &b.at(0, k * k_step));
//!     }
//!     // The MMAs of this stage, done once this resolves: wait on it before reusing the stage.
//!     let read = acc.commit();
//!     // ...
//! }
//! let acc = acc.wait();
//! ```
//!
//! Shared memory the units wrote is only visible to the MMAs after
//! [`sync_async_proxy_shared`](fn@crate::prelude::sync_async_proxy_shared) and a cube sync; a TMA
//! load needs neither.
//!
//! Requires [`wgmma`](cubecl_ir::features::MatmulFeatures::wgmma), only present on `sm_90a`.

mod layout;

pub use layout::{Major, Swizzle, WgmmaLayoutError, WgmmaTileLayout};

use super::pending::{GroupToken, PendingExpand};
use crate::{self as cubecl, prelude::*, unexpanded};
use core::marker::PhantomData;
use cubecl_ir::{
    dialect::matrix::{WgmmaDescriptorOp, WgmmaFenceOp, WgmmaFenceOperandOp, WgmmaOp},
    features::WgmmaElems,
    types::{MatrixShape, pending::AsyncGroup},
};
use cubecl_macros::{comptime_type, cube, intrinsic};
use pliron::builtin::attributes::TypeAttr;

use alloc::format;

/// Units in a warpgroup.
pub const WARPGROUP_UNITS: u32 = 128;
/// Rows every warpgroup MMA computes.
pub const WARPGROUP_M: usize = 64;

/// The 64-bit descriptor of a tile in shared memory, which a warpgroup MMA reads `A` or `B`
/// through.
#[derive(CubeType, Clone, Copy)]
pub struct MatrixDescriptor<E: Scalar> {
    #[allow(unused)]
    descriptor: u64,
    #[allow(unused)]
    #[cube(comptime)]
    layout: WgmmaTileLayout,
    #[cube(comptime)]
    _elem: PhantomData<E>,
}

#[cube]
impl<E: Scalar> MatrixDescriptor<E> {
    /// Describes `tile`, laid out as `layout`. `tile` must be in shared memory and start at a
    /// multiple of [`WgmmaTileLayout::alignment`]: a swizzled tile anywhere else is read with
    /// its lines permuted wrong, silently.
    #[allow(unused_variables)]
    pub fn new(tile: &[E], #[comptime] layout: WgmmaTileLayout) -> Self {
        intrinsic!(|scope| {
            if let Err(err) = layout.validate(E::elem_type(scope).size()) {
                scope.push_error(format!("a warpgroup MMA can't read the tile: {err}"));
            }
            let ptr = unsafe { *tile.__expand_as_ptr_method(scope) }.value(scope);
            let op = WgmmaDescriptorOp::new(
                scope.ctx_mut(),
                ptr,
                layout.leading_byte_offset(),
                layout.stride_byte_offset(),
                layout.swizzle,
            );
            MatrixDescriptorExpand {
                descriptor: scope.register_with_result(&op).into(),
                layout,
                _elem: PhantomData,
            }
        })
    }

    /// The descriptor of the sub-tile that starts at `(row, k)`, which an MMA reads its 64 rows
    /// of `A`, or its `n` rows of `B`, and [`WgmmaTileLayout::k_step`] of K from. `row` is a
    /// multiple of 8, and `k` of the step.
    pub fn at(&self, row: usize, k: usize) -> Self {
        let elem_size = E::size();
        let offset = comptime![self.layout].byte_offset(row, k, elem_size);
        // The address field holds the address in 16-byte units, and no shared memory address
        // carries out of it.
        MatrixDescriptor::<E> {
            descriptor: self.descriptor + (offset as u64 >> 4),
            layout: comptime![self.layout],
            _elem: PhantomData,
        }
    }

    /// The layout of the tile.
    pub fn layout(&self) -> comptime_type!(WgmmaTileLayout) {
        intrinsic!(|_| self.layout)
    }
}

/// The registers each unit of a warpgroup holds of a `64 x n` accumulator.
#[derive(CubeType)]
pub struct Accumulator<CD: Numeric> {
    registers: Array<CD>,
    #[allow(unused)]
    #[cube(comptime)]
    n: usize,
}

#[cube]
impl<CD: Numeric> Accumulator<CD> {
    /// A zeroed `64 x n` accumulator. `n` is a multiple of 8 up to 256, or of 16 for the integer
    /// types.
    pub fn new(#[comptime] n: usize) -> Self {
        let len = comptime![WARPGROUP_M * n / WARPGROUP_UNITS as usize];
        let mut registers = Array::new(len);
        #[unroll]
        for i in 0..len {
            registers[i] = CD::from_int(0);
        }
        Accumulator::<CD> { registers, n }
    }

    /// The `n` of the accumulator.
    pub fn n(&self) -> comptime_type!(usize) {
        intrinsic!(|_| self.n)
    }

    /// The elements each unit holds.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> comptime_type!(usize) {
        intrinsic!(|_| WARPGROUP_M * self.n / WARPGROUP_UNITS as usize)
    }

    /// The `nth` element this unit holds.
    pub fn get(&self, nth: usize) -> CD {
        self.registers[nth]
    }

    /// The `(row, col)` of the `nth` element `unit` holds. `unit` is the unit's position in its
    /// warpgroup, `0..128`.
    ///
    /// Each plane holds 16 rows. A unit holds two adjacent elements of its row `lane / 4` of
    /// them, then the same in row `lane / 4 + 8`, then moves on 8 columns.
    pub fn position_of_nth(&self, unit: u32, nth: u32) -> (u32, u32) {
        position_of_nth(unit, nth, 2u32)
    }

    /// Hands the accumulator to the MMAs that [`Pending::execute`] issues.
    pub fn start(self) -> Pending<Accumulator<CD>> {
        intrinsic!(|scope| {
            let mut registers = self.registers;
            fence_operand(scope, &mut registers);
            scope.register(&WgmmaFenceOp::new(scope.ctx_mut()));
            PendingExpand::ready(scope, AsyncGroup::Warpgroup, self)
        })
    }
}

impl<CD: Numeric> PendingValue for Accumulator<CD> {
    fn wait_for(_pending: Pending<Self>) -> Self {
        unexpanded!()
    }

    /// Waits for every MMA issued into the accumulator, committed or not.
    fn __expand_wait_for(scope: &Scope, pending: PendingExpand<Self>) -> AccumulatorExpand<CD> {
        let token = GroupToken::commit(scope, AsyncGroup::Warpgroup);
        token.__expand_wait_method(scope);
        let mut registers = pending.value.registers;
        // The MMAs wrote the registers behind the compiler's back: no read moves above the wait.
        fence_operand(scope, &mut registers);
        pending.value
    }
}

#[cube]
impl<CD: Numeric> Pending<Accumulator<CD>> {
    /// Issues `D = A * B + D`, with `A` and `B` in shared memory. Each MMA reads 64 rows of `A`
    /// and `n` of `B`, and [`WgmmaTileLayout::k_step`] elements of K, from where the
    /// descriptors point.
    #[allow(unused_variables)]
    pub fn execute<A: Scalar, B: Scalar>(
        &mut self,
        a: &MatrixDescriptor<A>,
        b: &MatrixDescriptor<B>,
    ) {
        intrinsic!(|scope| {
            let a_major = a.layout.major;
            let a = a.descriptor.read_value(scope);
            issue::<A, B, CD>(scope, &mut self.value, a, a_major, b)
        })
    }

    /// Issues `D = A * B + D`, with `A` in registers and `B` in shared memory. The fragment
    /// must stay unchanged until the MMA's group completes.
    #[allow(unused_variables)]
    pub fn execute_registers<A: Scalar, B: Scalar>(
        &mut self,
        a: &Fragment<A>,
        b: &MatrixDescriptor<B>,
    ) {
        intrinsic!(|scope| {
            // The fragment was written by the units, and the MMA reads it out of the same
            // registers: settle the writes, and order them before the MMA.
            let mut fragment = a.registers;
            fence_operand(scope, &mut fragment);
            scope.register(&WgmmaFenceOp::new(scope.ctx_mut()));
            let a = a.registers.read_value(scope);
            issue::<A, B, CD>(scope, &mut self.value, a, Major::K, b)
        })
    }

    /// Commits the MMAs issued since the last commit into a group, and returns its completion.
    /// Once it resolves, the MMAs are done reading their tiles and fragments.
    pub fn commit(&mut self) -> Pending<()> {
        intrinsic!(|scope| PendingExpand::commit(scope, AsyncGroup::Warpgroup, ()))
    }
}

fn fence_operand<E: CubePrimitive>(scope: &Scope, registers: &mut NativeExpand<Array<E>>) {
    let registers = registers.__extract_list(scope);
    scope.register(&WgmmaFenceOperandOp::new(scope.ctx_mut(), registers))
}

fn issue<A: Scalar, B: Scalar, CD: Numeric>(
    scope: &Scope,
    acc: &mut AccumulatorExpand<CD>,
    a: cubecl_ir::pliron::value::Value,
    a_major: Major,
    b: &MatrixDescriptorExpand<B>,
) {
    let elems = WgmmaElems {
        a: A::elem_type(scope),
        b: B::elem_type(scope),
        cd: CD::elem_type(scope),
    };
    let shape = MatrixShape {
        m: WARPGROUP_M,
        n: acc.n,
        k: WgmmaTileLayout::k_step(elems.a.size()),
    };
    check_supported(scope, elems, shape);
    let b_major = b.layout.major;
    let b = b.descriptor.read_value(scope);
    let registers = acc.registers.__extract_list(scope);
    let a_ty = TypeAttr::new(A::__expand_as_type(scope));
    let b_ty = TypeAttr::new(B::__expand_as_type(scope));
    scope.register(&WgmmaOp::new(
        scope.ctx_mut(),
        a,
        b,
        registers,
        a_ty,
        b_ty,
        shape,
        a_major,
        b_major,
    ));
}

fn check_supported(scope: &Scope, elems: WgmmaElems, shape: MatrixShape) {
    let Some(props) = scope.state().device_properties.clone() else {
        return;
    };
    let wgmma = &props.features.matmul.wgmma;
    if !wgmma.iter().any(|config| config.matches(elems, shape)) {
        let WgmmaElems { a, b, cd } = elems;
        let MatrixShape { m, n, k } = shape;
        scope.push_error(format!(
            "the device doesn't support a {m}x{n}x{k} warpgroup MMA of {a:?} x {b:?} into \
             {cd:?}; supported configurations: {wgmma:?}"
        ));
    }
}

/// The registers each unit of a warpgroup holds of a `64 x k` tile of `A`, for
/// [`Pending::execute_registers`]. `k` is [`WgmmaTileLayout::k_step`].
#[derive(CubeType)]
pub struct Fragment<A: Scalar> {
    registers: Array<A>,
}

#[cube]
impl<A: Scalar> Fragment<A> {
    /// An uninitialized fragment.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let elem_size = A::size();
        let len =
            comptime![WARPGROUP_M * WgmmaTileLayout::k_step(elem_size) / WARPGROUP_UNITS as usize];
        Fragment::<A> {
            registers: Array::new(len),
        }
    }

    /// The elements each unit holds.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> comptime_type!(usize) {
        intrinsic!(|scope| {
            let k = WgmmaTileLayout::k_step(A::elem_type(scope).size());
            WARPGROUP_M * k / WARPGROUP_UNITS as usize
        })
    }

    /// The `nth` element this unit holds.
    pub fn get(&self, nth: usize) -> A {
        self.registers[nth]
    }

    /// Sets the `nth` element this unit holds.
    pub fn set(&mut self, nth: usize, value: A) {
        self.registers[nth] = value;
    }

    /// The `(row, col)` of the `nth` element `unit` holds, laid out as the accumulator, but in
    /// runs of 32 bits rather than of two elements.
    pub fn position_of_nth(&self, unit: u32, nth: u32) -> (u32, u32) {
        let elem_size = A::size();
        position_of_nth(unit, nth, comptime![(4 / elem_size) as u32])
    }
}

/// The `(row, col)` of the `nth` element `unit` holds, in runs of `run` adjacent elements.
#[cube]
fn position_of_nth(unit: u32, nth: u32, #[comptime] run: u32) -> (u32, u32) {
    let plane = unit / 32;
    let lane = unit % 32;
    let chunk = nth / run;
    let row = plane * 16 + lane / 4 + (chunk % 2) * 8;
    let col = (chunk / 2) * (run * 4) + (lane % 4) * run + nth % run;
    (row, col)
}
