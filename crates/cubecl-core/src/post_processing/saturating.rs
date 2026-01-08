use crate as cubecl;
use cubecl_ir::{
    Allocator, Arithmetic, ElemType, ExpandElement, Instruction, IntKind, Operation, Processor,
    Scope, ScopeProcessing, StorageType, UIntKind, Variable,
};

use crate::prelude::*;

/// Replaces saturating arithmetic with a performant polyfill
#[derive(new, Debug)]
pub struct SaturatingArithmeticProcessor {
    /// Whether to replace i32 saturating sub. Used for CUDA, because there's a more performant
    /// PTX intrinsic for that specific type.
    replace_i32: bool,
}

impl Processor for SaturatingArithmeticProcessor {
    fn transform(
        &self,
        mut processing: cubecl_ir::ScopeProcessing,
        allocator: Allocator,
    ) -> cubecl_ir::ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            if let Operation::Arithmetic(arithmetic) = &instruction.operation {
                match arithmetic {
                    Arithmetic::SaturatingAdd(op) if op.lhs.elem_type().is_unsigned_int() => {
                        run_polyfill(
                            &mut processing,
                            op.lhs,
                            op.rhs,
                            instruction.out(),
                            &allocator,
                            saturating_add_unsigned::expand::<IntExpand<0>>,
                        );
                        continue;
                    }
                    Arithmetic::SaturatingAdd(op)
                        if op.lhs.elem_type().is_signed_int()
                            && self.should_replace(op.lhs.storage_type()) =>
                    {
                        run_polyfill(
                            &mut processing,
                            op.lhs,
                            op.rhs,
                            instruction.out(),
                            &allocator,
                            saturating_add_signed::expand::<IntExpand<0>, IntExpand<1>>,
                        );
                        continue;
                    }
                    Arithmetic::SaturatingSub(op) if op.lhs.elem_type().is_unsigned_int() => {
                        run_polyfill(
                            &mut processing,
                            op.lhs,
                            op.rhs,
                            instruction.out(),
                            &allocator,
                            saturating_sub_unsigned::expand::<IntExpand<0>>,
                        );
                        continue;
                    }
                    Arithmetic::SaturatingSub(op)
                        if op.lhs.elem_type().is_signed_int()
                            && self.should_replace(op.lhs.storage_type()) =>
                    {
                        run_polyfill(
                            &mut processing,
                            op.lhs,
                            op.rhs,
                            instruction.out(),
                            &allocator,
                            saturating_sub_signed::expand::<IntExpand<0>, IntExpand<1>>,
                        );
                        continue;
                    }
                    _ => {}
                }
            }

            // When we have nothing to do.
            processing.instructions.push(instruction);
        }
        processing
    }
}

impl SaturatingArithmeticProcessor {
    fn should_replace(&self, ty: StorageType) -> bool {
        self.replace_i32 || !matches!(ty, StorageType::Scalar(ElemType::Int(IntKind::I32)))
    }
}

fn run_polyfill<T: CubePrimitive>(
    processing: &mut ScopeProcessing,
    lhs: Variable,
    rhs: Variable,
    out: Variable,
    allocator: &Allocator,
    mut polyfill: impl FnMut(
        &mut Scope,
        ExpandElementTyped<T>,
        ExpandElementTyped<T>,
    ) -> ExpandElementTyped<T>,
) {
    let lhs = ExpandElement::Plain(lhs);
    let rhs = ExpandElement::Plain(rhs);
    let mut scope = Scope::root(false)
        .with_allocator(allocator.clone())
        .with_types(processing.typemap.clone());
    scope.register_type::<IntExpand<0>>(lhs.storage_type());
    if let ElemType::Int(kind) = lhs.elem_type() {
        let unsigned_ty = match kind {
            IntKind::I8 => UIntKind::U8,
            IntKind::I16 => UIntKind::U16,
            IntKind::I32 => UIntKind::U32,
            IntKind::I64 => UIntKind::U64,
        };
        scope.register_type::<IntExpand<1>>(ElemType::UInt(unsigned_ty).into())
    }

    let out_poly = polyfill(&mut scope, lhs.into(), rhs.into()).expand;
    let tmp_processing = scope.process([]);

    for inst in tmp_processing.instructions {
        processing.instructions.push(inst);
    }
    for var in tmp_processing.variables {
        processing.variables.push(var);
    }

    processing
        .instructions
        .push(Instruction::new(Operation::Copy(*out_poly), out));
}

#[cube]
fn saturating_add_unsigned<U: Int>(a: Line<U>, b: Line<U>) -> Line<U> {
    let c = a.min(!b);
    c + b
}

#[cube]
fn saturating_sub_unsigned<U: Int>(a: Line<U>, b: Line<U>) -> Line<U> {
    let a = a.max(b);
    a - b
}

/// Don't ask me how this works
/// <https://locklessinc.com/articles/sat_arithmetic/>
#[cube]
fn saturating_add_signed<I: Int, U: Int>(x: Line<I>, y: Line<I>) -> Line<I> {
    let bit_width = I::type_size_bits();
    let shift = Line::<U>::new(U::new(comptime![(bit_width - 1) as i64]));

    let ux = Line::<U>::cast_from(x);
    let uy = Line::<U>::cast_from(y);
    let res = ux + uy;
    let ux = (ux >> shift) + Line::<U>::cast_from(I::max_value());
    let cond = Line::<I>::cast_from((ux ^ uy) | !(uy ^ res)).greater_equal(Line::new(I::new(0)));
    select_many(cond, Line::cast_from(ux), Line::cast_from(res))
}

/// Don't ask me how this works
/// <https://locklessinc.com/articles/sat_arithmetic/>
#[cube]
fn saturating_sub_signed<I: Int, U: Int>(x: Line<I>, y: Line<I>) -> Line<I> {
    let bit_width = I::type_size_bits();
    let shift = Line::<U>::new(U::new(comptime![(bit_width - 1) as i64]));

    let ux = Line::<U>::cast_from(x);
    let uy = Line::<U>::cast_from(y);
    let res = ux - uy;
    let ux = (ux >> shift) + Line::<U>::cast_from(I::max_value());
    let cond = Line::<I>::cast_from((ux ^ uy) & (ux ^ res)).less_than(Line::new(I::new(0)));
    select_many(cond, Line::cast_from(ux), Line::cast_from(res))
}
