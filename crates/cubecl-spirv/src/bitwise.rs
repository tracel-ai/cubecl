use cubecl_core::{self as cubecl, ir::ExpandElement};
use cubecl_core::{comptime, ir as core, prelude::*};
use cubecl_core::{cube, ir::Bitwise};

use crate::{item::Elem, SpirvCompiler, SpirvTarget};

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

            //These operations only support 32 bit
            Bitwise::CountOnes(op) => {
                let item = self.compile_item(op.input.item);
                match item.elem() {
                    Elem::Int(64, _) => {
                        let mut scope = self.opt.root_scope.child();
                        scope.register_elem::<IntExpand<0>>(op.input.item.elem);
                        u64_count_bits::expand::<IntExpand<0>>(
                            &mut scope,
                            ExpandElement::Plain(op.input).into(),
                            ExpandElement::Plain(out).into(),
                        );
                        self.compile_polyfill(scope);
                    }
                    _ => {
                        self.compile_unary_op_cast(op, out, uniform, |b, _, ty, input, out| {
                            b.bit_count(ty, Some(out), input).unwrap();
                        });
                    }
                }
            }
            Bitwise::ReverseBits(op) => {
                let item = self.compile_item(op.input.item);
                match item.elem() {
                    Elem::Int(64, _) => {
                        let mut scope = self.opt.root_scope.child();
                        scope.register_elem::<IntExpand<0>>(op.input.item.elem);
                        let input = ExpandElement::Plain(op.input);
                        let out = ExpandElement::Plain(out);
                        u64_reverse::expand::<IntExpand<0>>(&mut scope, input.into(), out.into());
                        self.compile_polyfill(scope);
                    }
                    Elem::Int(32, _) => {
                        self.compile_unary_op(op, out, |b, _, ty, input, out| {
                            b.bit_reverse(ty, Some(out), input).unwrap();
                        });
                    }
                    Elem::Int(width, _) => {
                        let mut scope = self.opt.root_scope.child();
                        scope.register_elem::<IntExpand<0>>(op.input.item.elem);
                        small_int_reverse::expand::<IntExpand<0>>(
                            &mut scope,
                            ExpandElement::Plain(op.input).into(),
                            ExpandElement::Plain(out).into(),
                            width,
                        );
                        println!("{scope:#?}");
                        self.compile_polyfill(scope);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[cube]
pub(crate) fn small_int_reverse<I: Int>(x: Line<I>, out: &mut Line<I>, #[comptime] width: u32) {
    let shift = comptime!(32 - width);

    let reversed = Line::reverse_bits(Line::<u32>::cast_from(x));
    *out = Line::cast_from(reversed >> Line::new(shift))
}

#[cube]
pub(crate) fn u64_reverse<I: Int>(x: Line<I>, out: &mut Line<I>) {
    let shift = Line::new(I::new(32));

    let low = Line::<u32>::cast_from(x);
    let high = Line::<u32>::cast_from(x >> shift);

    let low_rev = Line::reverse_bits(low);
    let high_rev = Line::reverse_bits(high);
    // Swap low and high values
    let high = Line::cast_from(low_rev) << shift;
    *out = high | Line::cast_from(high_rev);
}

#[cube]
pub(crate) fn u64_count_bits<I: Int>(x: Line<I>, out: &mut Line<u32>) {
    let shift = Line::new(I::new(32));

    let low = Line::<u32>::cast_from(x);
    let high = Line::<u32>::cast_from(x >> shift);

    let low_cnt = Line::<u32>::cast_from(Line::count_ones(low));
    let high_cnt = Line::<u32>::cast_from(Line::count_ones(high));
    *out = low_cnt + high_cnt;
}
