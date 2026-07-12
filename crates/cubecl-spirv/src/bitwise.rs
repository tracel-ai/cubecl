// use cubecl_core::cube;
// use cubecl_core::{self as cubecl, ir::ElemType};
// use cubecl_core::{comptime, ir as core, prelude::*};

// use crate::{SpirvCompiler, SpirvTarget, item::Elem};

// impl<T: SpirvTarget> SpirvCompiler<T> {
//     pub fn compile_bitwise(&mut self, op: Bitwise, out: Option<core::ExpandValue>, uniform: bool) {
//         if let Some(op) = bool_op(&op) {
//             self.compile_operator(op, out, uniform);
//             return;
//         }

//         let out = out.unwrap();
//         match op {
//             Bitwise::LeadingZeros(op) => {
//                 let width = op.input.ty.storage_type().size() as u32 * 8;
//                 self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
//                     // Indices are zero based, so subtract 1
//                     let width = out_ty.const_u32(b, width - 1);
//                     let msb = b.id();
//                     T::find_msb(b, ty, input, msb);
//                     b.mark_uniformity(msb, uniform);
//                     b.i_sub(ty, Some(out), width, msb).unwrap();
//                 });
//             }
//             Bitwise::FindFirstSet(op) => {
//                 self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
//                     let one = out_ty.const_u32(b, 1);
//                     let lsb = b.id();
//                     T::find_lsb(b, ty, input, lsb);
//                     b.mark_uniformity(lsb, uniform);
//                     // Normalize to CUDA/POSIX convention of 1 based index, with 0 meaning not found
//                     b.i_add(ty, Some(out), lsb, one).unwrap();
//                 });
//             }
//             Bitwise::TrailingZeros(op) => {
//                 let width = op.input.ty.storage_type().size() as u32 * 8;
//                 self.compile_unary_op(op, out, uniform, |b, out_ty, ty, input, out| {
//                     // find_lsb returns -1 (0xFFFFFFFF) for zero input
//                     // trailing_zeros should return bit_width for zero input
//                     let width_const = out_ty.const_u32(b, width);
//                     let zero = out_ty.const_u32(b, 0);
//                     let lsb = b.id();
//                     T::find_lsb(b, ty, input, lsb);
//                     b.mark_uniformity(lsb, uniform);
//                     // Check if input is zero
//                     let bool_ty = out_ty.same_vectorization(Elem::Bool).id(b);
//                     let is_zero = b.id();
//                     b.i_equal(bool_ty, Some(is_zero), input, zero).unwrap();
//                     b.mark_uniformity(is_zero, uniform);
//                     // Select width if zero, otherwise lsb
//                     b.select(ty, Some(out), is_zero, width_const, lsb).unwrap();
//                 });
//             }
//         }
//     }
// }

// #[cube]
// pub(crate) fn u64_reverse<I: Int, N: Size>(x: Vector<I, N>) -> Vector<I, N> {
//     let shift = Vector::new(I::new(32));

//     let low = Vector::<u32, N>::cast_from(x);
//     let high = Vector::<u32, N>::cast_from(x >> shift);

//     let low_rev = Vector::reverse_bits(low);
//     let high_rev = Vector::reverse_bits(high);
//     // Swap low and high values
//     let high = Vector::cast_from(low_rev) << shift;
//     high | Vector::cast_from(high_rev)
// }

// #[cube]
// pub(crate) fn u64_count_bits<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
//     let shift = Vector::new(I::new(32));

//     let low = Vector::<u32, N>::cast_from(x);
//     let high = Vector::<u32, N>::cast_from(x >> shift);

//     let low_cnt = Vector::<u32, N>::cast_from(Vector::count_ones(low));
//     let high_cnt = Vector::<u32, N>::cast_from(Vector::count_ones(high));
//     low_cnt + high_cnt
// }

// #[cube]
// pub(crate) fn u64_leading_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
//     let shift = Vector::new(I::new(32));

//     let low = Vector::<u32, N>::cast_from(x);
//     let high = Vector::<u32, N>::cast_from(x >> shift);
//     let low_zeros = Vector::leading_zeros(low);
//     let high_zeros = Vector::leading_zeros(high);

//     select_many(
//         high_zeros.equal(&Vector::new(32)),
//         low_zeros + high_zeros,
//         high_zeros,
//     )
// }

// /// There are three possible outcomes:
// /// * low has any set -> return low
// /// * low is empty, high has any set -> return high + 32
// /// * low and high are empty -> return 0
// #[cube]
// pub(crate) fn u64_ffs<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
//     let shift = Vector::new(I::new(32));

//     let low = Vector::<u32, N>::cast_from(x);
//     let high = Vector::<u32, N>::cast_from(x >> shift);
//     let low_ffs = Vector::find_first_set(low);
//     let high_ffs = Vector::find_first_set(high);

//     let high_ffs = select_many(
//         high_ffs.equal(&Vector::new(0)),
//         high_ffs,
//         high_ffs + Vector::new(32),
//     );
//     select_many(low_ffs.equal(&Vector::new(0)), high_ffs, low_ffs)
// }

// /// Subtract extra leading zeros after normalizing
// #[cube]
// pub(crate) fn u16_u8_leading_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
//     let width = I::type_size_bits().comptime() as u32;
//     let over_width = Vector::new(32 - width);

//     let x = Vector::<u32, N>::cast_from(x);
//     let lz = x.leading_zeros();
//     lz - over_width
// }

// /// There are three possible outcomes:
// /// * low has any set -> return low
// /// * low is empty, high has any set -> return high + 32
// /// * low and high are empty -> return 0
// #[cube]
// pub(crate) fn u64_trailing_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
//     let shift = Vector::new(I::new(32));

//     let low = Vector::<u32, N>::cast_from(x);
//     let high = Vector::<u32, N>::cast_from(x >> shift);
//     let low_tz = Vector::trailing_zeros(low);
//     let high_tz = Vector::trailing_zeros(high);

//     let high_tz = select_many(
//         high_tz.equal(&Vector::new(32)),
//         Vector::new(64),
//         high_tz + Vector::new(32),
//     );
//     select_many(low_tz.equal(&Vector::new(32)), high_tz, low_tz)
// }

// /// Clamp to width
// #[cube]
// pub(crate) fn u16_u8_trailing_zeros<I: Int, N: Size>(x: Vector<I, N>) -> Vector<u32, N> {
//     let width = Vector::new(I::type_size_bits().comptime() as u32);

//     let x = Vector::<u32, N>::cast_from(x);
//     let lz = x.trailing_zeros();
//     lz.min(width)
// }
