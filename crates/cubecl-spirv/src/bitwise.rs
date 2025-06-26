use cubecl_core as cubecl;
use cubecl_core::{comptime, ir as core, prelude::*};
use cubecl_core::{cube, ir::Bitwise};

use crate::{SpirvCompiler, SpirvTarget};

impl<T: SpirvTarget> SpirvCompiler<T> {
    pub fn compile_bitwise(&mut self, op: Bitwise, out: Option<core::Variable>, uniform: bool) {
        let out = out.unwrap();
        match op {
            Bitwise::BitwiseAnd(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_and(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::BitwiseOr(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_or(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::BitwiseXor(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.bitwise_xor(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::BitwiseNot(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, _, ty, input, out| {
                    b.not(ty, Some(out), input).unwrap();
                });
            }
            Bitwise::ShiftLeft(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.shift_left_logical(ty, Some(out), lhs, rhs).unwrap();
                })
            }
            Bitwise::ShiftRight(op) => {
                self.compile_binary_op(op, out, uniform, |b, _, ty, lhs, rhs, out| {
                    b.shift_right_logical(ty, Some(out), lhs, rhs).unwrap();
                })
            }

            Bitwise::CountOnes(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, _, ty, input, out| {
                    b.bit_count(ty, Some(out), input).unwrap();
                });
            }
            Bitwise::ReverseBits(op) => {
                self.compile_unary_op(op, out, uniform, |b, _, ty, input, out| {
                    b.bit_reverse(ty, Some(out), input).unwrap();
                });
            }
            Bitwise::LeadingZeros(op) => {
                let width = op.input.item.elem.size() as u32 * 8;
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    // Indices are zero based, so subtract 1
                    let width = out_ty.const_u32(b, width - 1);
                    let msb = b.id();
                    T::find_msb(b, ty, input, msb);
                    b.mark_uniformity(msb, uniform);
                    b.i_sub(ty, Some(out), width, msb).unwrap();
                });
            }
            Bitwise::FindFirstSet(op) => {
                self.compile_unary_op_cast(op, out, uniform, |b, out_ty, ty, input, out| {
                    let one = out_ty.const_u32(b, 1);
                    let lsb = b.id();
                    T::find_lsb(b, ty, input, lsb);
                    b.mark_uniformity(lsb, uniform);
                    // Normalize to CUDA/POSIX convention of 1 based index, with 0 meaning not found
                    b.i_add(ty, Some(out), lsb, one).unwrap();
                });
            }
        }
    }
}

#[cube]
pub(crate) fn small_int_reverse<I: Int>(x: Line<I>, #[comptime] width: u32) -> Line<I> {
    let shift = comptime!(32 - width);

    let reversed = Line::reverse_bits(Line::<u32>::cast_from(x));
    Line::cast_from(reversed >> Line::new(shift))
}

#[cube]
pub(crate) fn u64_reverse<I: Int>(x: Line<I>) -> Line<I> {
    let shift = Line::new(I::new(32));

    let low = Line::<u32>::cast_from(x);
    let high = Line::<u32>::cast_from(x >> shift);

    let low_rev = Line::reverse_bits(low);
    let high_rev = Line::reverse_bits(high);
    // Swap low and high values
    let high = Line::cast_from(low_rev) << shift;
    high | Line::cast_from(high_rev)
}

#[cube]
pub(crate) fn u64_count_bits<I: Int>(x: Line<I>) -> Line<u32> {
    let shift = Line::new(I::new(32));

    let low = Line::<u32>::cast_from(x);
    let high = Line::<u32>::cast_from(x >> shift);

    let low_cnt = Line::<u32>::cast_from(Line::count_ones(low));
    let high_cnt = Line::<u32>::cast_from(Line::count_ones(high));
    low_cnt + high_cnt
}

#[cube]
pub(crate) fn u64_leading_zeros<I: Int>(x: Line<I>) -> Line<u32> {
    let shift = Line::new(I::new(32));

    let low = Line::<u32>::cast_from(x);
    let high = Line::<u32>::cast_from(x >> shift);
    let low_zeros = Line::leading_zeros(low);
    let high_zeros = Line::leading_zeros(high);

    select_many(
        high_zeros.equal(Line::new(32)),
        low_zeros + high_zeros,
        high_zeros,
    )
}

/// There are three possible outcomes:
/// * low has any set -> return low
/// * low is empty, high has any set -> return high + 32
/// * low and high are empty -> return 0
#[cube]
pub(crate) fn u64_ffs<I: Int>(x: Line<I>) -> Line<u32> {
    let shift = Line::new(I::new(32));

    let low = Line::<u32>::cast_from(x);
    let high = Line::<u32>::cast_from(x >> shift);
    let low_ffs = Line::find_first_set(low);
    let high_ffs = Line::find_first_set(high);

    let high_ffs = select_many(
        high_ffs.equal(Line::new(0)),
        high_ffs,
        high_ffs + Line::new(32),
    );
    select_many(low_ffs.equal(Line::new(0)), high_ffs, low_ffs)
}
