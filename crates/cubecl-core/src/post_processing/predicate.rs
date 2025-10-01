use core::{f32, f64};

use crate as cubecl;
use cubecl_ir::{
    Allocator, Comparison, ElemType, ExpandElement, FloatKind, Instruction, Operation, Processor,
    Scope, ScopeProcessing, UIntKind, Variable,
};
use half::{bf16, f16};

use crate::prelude::*;

#[derive(Debug, Default)]
pub struct PredicateProcessor;

impl Processor for PredicateProcessor {
    fn transform(
        &self,
        mut processing: cubecl_ir::ScopeProcessing,
        allocator: Allocator,
    ) -> cubecl_ir::ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            if let Operation::Comparison(comparison) = &instruction.operation {
                match comparison {
                    Comparison::IsNan(op) => {
                        run_polyfill(
                            &mut processing,
                            op.input,
                            instruction.out(),
                            &allocator,
                            is_nan::expand::<FloatExpand<0>, IntExpand<1>>,
                        );
                        continue;
                    }
                    Comparison::IsInf(op) => {
                        run_polyfill(
                            &mut processing,
                            op.input,
                            instruction.out(),
                            &allocator,
                            is_inf::expand::<FloatExpand<0>, IntExpand<1>>,
                        );
                        continue;
                    }
                    _ => {}
                }
            }
            processing.instructions.push(instruction);
        }
        processing
    }
}

fn run_polyfill<T: CubePrimitive, O: CubePrimitive>(
    processing: &mut ScopeProcessing,
    input: Variable,
    out: Variable,
    allocator: &Allocator,
    mut polyfill: impl FnMut(&mut Scope, ExpandElementTyped<T>, u32, u32) -> ExpandElementTyped<O>,
) {
    let input = ExpandElement::Plain(input);
    let mut scope = Scope::root(false).with_allocator(allocator.clone());
    scope.register_type::<FloatExpand<0>>(input.storage_type());

    let out_poly = if let ElemType::Float(kind) = input.elem_type() {
        let (unsigned_ty, bit_width, mantissa_bits) = match kind {
            FloatKind::F64 => (
                UIntKind::U64,
                f64::size_bits().unwrap(),
                f64::MANTISSA_DIGITS - 1,
            ),
            FloatKind::F32 => (
                UIntKind::U32,
                f32::size_bits().unwrap(),
                f32::MANTISSA_DIGITS - 1,
            ),
            FloatKind::F16 => (
                UIntKind::U16,
                f16::size_bits().unwrap(),
                f16::MANTISSA_DIGITS - 1,
            ),
            FloatKind::BF16 => (
                UIntKind::U16,
                bf16::size_bits().unwrap(),
                bf16::MANTISSA_DIGITS - 1,
            ),
            _ => unreachable!(),
        };
        scope.register_type::<IntExpand<1>>(ElemType::UInt(unsigned_ty).into());

        let exp_bits = bit_width as u32 - mantissa_bits - 1;

        polyfill(&mut scope, input.into(), mantissa_bits, exp_bits).expand
    } else {
        panic!("Should be float")
    };

    let tmp_processing = scope.process([]);

    processing.instructions.extend(tmp_processing.instructions);
    processing.variables.extend(tmp_processing.variables);

    processing
        .instructions
        .push(Instruction::new(Operation::Copy(*out_poly), out));
}

#[cube]
fn is_nan<F: Float, U: Int>(
    x: Line<F>,
    #[comptime] mantissa_bits: u32,
    #[comptime] exp_bits: u32,
) -> Line<bool> {
    // Need to mark as i64 otherwise it is coerced into i32 which does not fit the values for f64
    let inf_bits = comptime![((1i64 << exp_bits as i64) - 1i64) << mantissa_bits as i64];
    let abs_mask = comptime![(1i64 << (exp_bits as i64 + mantissa_bits as i64)) - 1i64];

    let bits: Line<U> = Line::<U>::reinterpret(x);

    let abs_bits = bits & Line::new(U::new(abs_mask));

    abs_bits.greater_than(Line::new(U::new(inf_bits)))
}

// Same trick as NaN detection following IEEE 754, but check for all 0 bits equality
#[cube]
fn is_inf<F: Float, U: Int>(
    x: Line<F>,
    #[comptime] mantissa_bits: u32,
    #[comptime] exp_bits: u32,
) -> Line<bool> {
    // Need to mark as i64 otherwise it is coerced into i32 which does not fit the values for f64
    let inf_bits = comptime![((1i64 << exp_bits as i64) - 1i64) << mantissa_bits as i64];
    let abs_mask = comptime![(1i64 << (exp_bits as i64 + mantissa_bits as i64)) - 1i64];

    let bits: Line<U> = Line::<U>::reinterpret(x);

    let abs_bits = bits & Line::new(U::new(abs_mask));

    abs_bits.equal(Line::new(U::new(inf_bits)))
}
