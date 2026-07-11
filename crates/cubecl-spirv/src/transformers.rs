// use cubecl_core::{
//     define_scalar, define_size,
//     ir::{ElemType, ExpandValue, IntKind, Scope, Type, UIntKind},
//     prelude::{assign, expand_erf, expand_hypot, expand_rhypot},
// };

// use crate::bitwise::{
//     small_int_reverse, u16_u8_leading_zeros, u16_u8_trailing_zeros, u64_count_bits, u64_ffs,
//     u64_leading_zeros, u64_reverse, u64_trailing_zeros,
// };

// define_scalar!(IntA);
// define_size!(SizeA);

// /// Transform operations that only support 32 bits using polyfills
// #[derive(Debug)]
// pub(crate) struct BitwiseTransform {
//     /// Allow base (non-extension) instructions with arbitrary bit widths. As far as I can tell,
//     /// extension functions are still limited with maintenance9.
//     arbitrary_bitwise: bool,
// }

// impl BitwiseTransform {
//     pub(crate) fn new(arbitrary_bitwise: bool) -> Self {
//         Self { arbitrary_bitwise }
//     }
// }

// impl IrTransformer for BitwiseTransform {
//     fn maybe_transform(&self, scope: &Scope, inst: &Instruction) -> TransformAction {
//         let op = match &inst.operation {
//             Operation::Bitwise(op) => op,
//             _ => return TransformAction::Ignore,
//         };
//         match op {
//             Bitwise::TrailingZeros(op) if is_u64(op.input) => {
//                 let scope = scope.child();
//                 scope.register_type::<IntA>(op.input.storage_type());
//                 scope.register_size::<SizeA>(op.input.vector_size());
//                 let res = u64_trailing_zeros::expand::<IntA, SizeA>(&scope, op.input.into());
//                 assign::expand_no_check(&scope, res, &mut inst.out().into());
//                 TransformAction::Replace(into_instructions(scope))
//             }
//             Bitwise::TrailingZeros(op) if is_u16_u8(op.input) => {
//                 let scope = scope.child();
//                 scope.register_type::<IntA>(op.input.storage_type());
//                 scope.register_size::<SizeA>(op.input.vector_size());
//                 let res = u16_u8_trailing_zeros::expand::<IntA, SizeA>(&scope, op.input.into());
//                 assign::expand_no_check(&scope, res, &mut inst.out().into());
//                 TransformAction::Replace(into_instructions(scope))
//             }
//             _ => TransformAction::Ignore,
//         }
//     }
// }

// fn is_u64(val: ExpandValue) -> bool {
//     matches!(
//         val.ty.elem_type(),
//         ElemType::Int(IntKind::I64) | ElemType::UInt(UIntKind::U64)
//     )
// }

// fn is_u16_u8(val: ExpandValue) -> bool {
//     matches!(
//         val.ty.elem_type(),
//         ElemType::Int(IntKind::I16)
//             | ElemType::UInt(UIntKind::U16)
//             | ElemType::Int(IntKind::I8)
//             | ElemType::UInt(UIntKind::U8)
//     )
// }
