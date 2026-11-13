//! Quantized MMA: a plane multiplies operands as their quantization scheme stores them, codes
//! under block scales, and sums the product into `f32` under its two scales.
//!
//! An `m × n × k` tile lies within one block of `block_k` values of `k`, so each row of `A` and
//! each column of `B` has one scale over it, and a lane's accumulator element at `(i, j)` becomes
//!
//! ```text
//! d[i][j] = c[i][j] + sa[i] · sb[j] · Σ_l qa[i][l] · qb[l][j]
//! ```
//!
//! where `qa` and `qb` are the codes' values and `sa`, `sb` the decoded block scales. A block
//! longer than the tile is several tiles under the same scales, and a per-tensor scale above the
//! block ones is the caller's to apply. The device lowers it to a block-scaled instruction where
//! it has one, and otherwise to its plain MMA over the codes.

use super::{MatrixIdent, MatrixLayout, MmaDefinitionExpand};
use crate::{self as cubecl, prelude::*};
use core::marker::PhantomData;
use cubecl_ir::{
    ElemType, FloatKind, Scope, VectorSize, dialect::matrix::MmaQuantizedOp,
    features::QuantizedMmaConfig,
};
use cubecl_macros::{comptime_type, cube, intrinsic};
use pliron::r#type::TypeHandle;

/// A quantized MMA the device lowers, of the [`QuantizedMmaConfig`] it advertises: `A` and `B`
/// are the codes each operand's [`QuantValue`](cubecl_common::quant::scheme::QuantValue) is held
/// in, `S` the type its [`ScaleDtype`](cubecl_common::quant::scheme::ScaleDtype) is stored in.
///
/// The codes are laid out in registers as the plain MMA of the same shape lays its operands out:
/// [`position_of_nth`](Self::position_of_nth) places them. The scales come one pair per
/// accumulator element a lane holds: [`scale_index_of_nth`](Self::scale_index_of_nth) places
/// them.
#[derive(Copy, Clone)]
pub struct QuantizedMma<A: CubeType, B: CubeType, S: CubeType> {
    _a: PhantomData<A>,
    _b: PhantomData<B>,
    _s: PhantomData<S>,
}

/// Expand type of [`QuantizedMma`].
#[derive(Debug)]
pub struct QuantizedMmaExpand<A: CubeType, B: CubeType, S: CubeType> {
    /// The plain MMA over the codes, whose register layout this one shares.
    codes: MmaDefinitionExpand<A, B, f32>,
    config: QuantizedMmaConfig,
    _s: PhantomData<S>,
}

impl<A: CubeType, B: CubeType, S: CubeType> Copy for QuantizedMmaExpand<A, B, S> {}
impl<A: CubeType, B: CubeType, S: CubeType> Clone for QuantizedMmaExpand<A, B, S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: CubeType, B: CubeType, S: CubeType> ExpandTypeClone for QuantizedMmaExpand<A, B, S> {
    fn clone_unchecked(&self) -> Self {
        *self
    }
}

impl<A: CubeType, B: CubeType, S: CubeType> AsRefExpand for QuantizedMmaExpand<A, B, S> {
    fn __expand_ref_method(&self, _scope: &Scope) -> &Self {
        self
    }
}

impl<A: CubeType, B: CubeType, S: CubeType> AsMutExpand for QuantizedMmaExpand<A, B, S> {
    fn __expand_ref_mut_method(&mut self, _scope: &Scope) -> &mut Self {
        self
    }
}

impl<A: CubeType, B: CubeType, S: CubeType> CubeType for QuantizedMma<A, B, S> {
    type ExpandType = QuantizedMmaExpand<A, B, S>;
}

impl<A: CubeType, B: CubeType, S: CubeType> IntoExpand for QuantizedMmaExpand<A, B, S> {
    type Expand = Self;

    fn into_expand(self, _: &Scope) -> Self::Expand {
        self
    }
}

impl<A: CubeType, B: CubeType, S: CubeType> IntoMut for QuantizedMmaExpand<A, B, S> {
    fn into_mut(self, _scope: &Scope) -> Self {
        self
    }
}

impl<A: CubeType, B: CubeType, S: CubeType> CubeDebug for QuantizedMmaExpand<A, B, S> {}

/// Panics unless `held` is the type `named` names: a quantized MMA's register types are the ones
/// its configuration's quantization types are held in.
fn assert_held_in(scope: &Scope, held: TypeHandle, named: ElemType, what: &str) {
    let expected = named.to_type(scope.ctx());
    assert!(
        held == expected,
        "QuantizedMma: the {what} are held in {named:?}, not the type the kernel names"
    );
}

#[cube]
impl<A: Scalar, B: Scalar, S: Scalar> QuantizedMma<A, B, S> {
    /// The quantized MMA of `config`, one the device advertises in
    /// [`quantized_mma`](cubecl_ir::features::MatmulFeatures::quantized_mma).
    ///
    /// Panics unless `A`, `B` and `S` are the types `config`'s codes and scales are held in: the
    /// [`ElemType::from_quant_value`] of each operand's value and the
    /// [`ElemType::from_scale_dtype`] of its scales.
    pub fn new(#[comptime] config: QuantizedMmaConfig) -> Self {
        intrinsic!(|scope| {
            let a_type = A::Scalar::__expand_as_type(scope);
            let b_type = B::Scalar::__expand_as_type(scope);
            let scales_type = S::Scalar::__expand_as_type(scope);
            assert_held_in(
                scope,
                a_type,
                ElemType::from_quant_value(config.a),
                "A codes",
            );
            assert_held_in(
                scope,
                b_type,
                ElemType::from_quant_value(config.b),
                "B codes",
            );
            let scales = ElemType::from_scale_dtype(config.scales);
            assert_held_in(scope, scales_type, scales, "scales");
            let shape = (config.m as usize, config.n as usize, config.k as usize).into();
            let codes = MmaDefinitionExpand {
                shape,
                a_type,
                b_type,
                cd_type: ElemType::Float(FloatKind::F32).to_type(scope.ctx()),
                scales_factor: None,
                scales_type: None,
                _a: PhantomData,
                _b: PhantomData,
                _cd: PhantomData,
            };
            QuantizedMmaExpand {
                codes,
                config,
                _s: PhantomData,
            }
        })
    }

    /// Elements of role `ident` each lane holds: codes for `A` and `B`, sums for the
    /// accumulator.
    pub fn elems_per_lane(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|scope| self.codes.__expand_elems_per_lane_method(scope, ident))
    }

    /// Elements of role `ident` held in each vector of registers.
    pub fn vector_size(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(VectorSize) {
        intrinsic!(|scope| self.codes.__expand_vector_size_method(scope, ident))
    }

    /// Vectors of [`vector_size`](Self::vector_size) each lane holds of role `ident`.
    pub fn vectors_per_lane(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(usize) {
        intrinsic!(|scope| self.codes.__expand_vectors_per_lane_method(scope, ident))
    }

    /// The layout of each vector of role `ident`.
    pub fn vector_layout(&self, #[comptime] ident: MatrixIdent) -> comptime_type!(MatrixLayout) {
        intrinsic!(|scope| self.codes.__expand_vector_layout_method(scope, ident))
    }

    /// The `(row, col)` of the `nth` element of role `ident` lane `lane_id` holds.
    pub fn position_of_nth(
        &self,
        lane_id: u32,
        nth: u32,
        #[comptime] ident: MatrixIdent,
    ) -> (u32, u32) {
        intrinsic!(|scope| self
            .codes
            .__expand_position_of_nth_method(scope, lane_id, nth, ident))
    }

    /// Scales each lane holds of either operand: one per accumulator element.
    pub fn scales_per_lane(&self) -> comptime_type!(usize) {
        self.elems_per_lane(MatrixIdent::Accumulator)
    }

    /// Which of its operand's scales the `nth` scale of role `ident` lane `lane_id` holds is:
    /// the row of `A`, or the column of `B`, of the lane's `nth` accumulator element.
    pub fn scale_index_of_nth(
        &self,
        lane_id: u32,
        nth: u32,
        #[comptime] ident: MatrixIdent,
    ) -> u32 {
        comptime!(assert!(
            ident != MatrixIdent::Accumulator,
            "QuantizedMma: the accumulator has no scales"
        ));
        let (row, col) = self.position_of_nth(lane_id, nth, MatrixIdent::Accumulator);
        match ident {
            MatrixIdent::A => row,
            _ => col,
        }
    }

    /// Multiplies `codes_a` by `codes_b` under `scales_a` and `scales_b`, summing into `acc`.
    /// Registers are laid out as [`position_of_nth`](Self::position_of_nth) and
    /// [`scale_index_of_nth`](Self::scale_index_of_nth) place them, the scales
    /// [`scales_per_lane`](Self::scales_per_lane) long, as they are stored.
    pub fn execute<NA: Size, NB: Size, NC: Size>(
        &self,
        codes_a: &Array<Vector<A, NA>>,
        codes_b: &Array<Vector<B, NB>>,
        scales_a: &Array<S>,
        scales_b: &Array<S>,
        acc: &Array<Vector<f32, NC>>,
    ) -> Array<Vector<f32, NC>> {
        // Decoded here, as any cast is, so every target's codec applies to every scale type.
        let count = self.scales_per_lane();
        let mut decoded_a = Array::<f32>::new(count);
        let mut decoded_b = Array::<f32>::new(count);
        #[unroll]
        for i in 0..count {
            decoded_a[i] = f32::cast_from(scales_a[i]);
            decoded_b[i] = f32::cast_from(scales_b[i]);
        }
        self.execute_decoded(codes_a, codes_b, &decoded_a, &decoded_b, acc)
    }

    /// [`execute`](Self::execute) with the scales decoded to `f32`.
    fn execute_decoded<NA: Size, NB: Size, NC: Size>(
        &self,
        codes_a: &Array<Vector<A, NA>>,
        codes_b: &Array<Vector<B, NB>>,
        scales_a: &Array<f32>,
        scales_b: &Array<f32>,
        acc: &Array<Vector<f32, NC>>,
    ) -> Array<Vector<f32, NC>> {
        intrinsic!(|scope| {
            let sums = self
                .codes
                .__expand_elems_per_lane_method(scope, MatrixIdent::Accumulator);
            let sum_vector = self
                .codes
                .__expand_vector_size_method(scope, MatrixIdent::Accumulator);
            let result = Array::__expand_new(scope, sums / sum_vector);

            let codes_a = codes_a.read_value(scope);
            let codes_b = codes_b.read_value(scope);
            let acc = acc.read_value(scope);
            let scales_a = scales_a.read_value(scope);
            let scales_b = scales_b.read_value(scope);
            let registers_d = result.__extract_list(scope);

            scope.register(&MmaQuantizedOp::new(
                scope.ctx_mut(),
                codes_a,
                codes_b,
                acc,
                registers_d,
                scales_a,
                scales_b,
                self.config.block_k as usize,
                self.codes.shape,
            ));

            result
        })
    }
}
