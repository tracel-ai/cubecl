// use cubecl_core::ir::{self};
// use tracel_llvm::mlir_rs::{
//     Context,
//     dialect::{arith, ods::vector},
//     ir::{
//         Type, Value,
//         attribute::{FloatAttribute, IntegerAttribute},
//         r#type::IntegerType,
//     },
// };

// use super::prelude::*;

// impl IntoType for ir::Type {
//     fn to_type<'a>(self, context: &'a Context) -> Type<'a> {
//         let inner_type = match self {
//             ir::Type::Opaque(ir::OpaqueType::Barrier(..)) => IntegerType::new(context, 32).into(),
//             other => other.storage_type().to_type(context),
//         };
//         match self.vector_size() {
//             size if size > 1 => Type::vector(&[size as u64], inner_type),
//             _ => inner_type,
//         }
//     }

//     fn is_vectorized(&self) -> bool {
//         self.vector_size() > 1
//     }
// }

// impl<'a> Visitor<'a> {
//     pub fn create_float_constant_from_item(&self, item: ir::Type, constant: f64) -> Value<'a, 'a> {
//         let float = item.storage_type().to_type(self.context);
//         let constant = FloatAttribute::new(self.context, float, constant);
//         let constant = self.append_operation_with_result(arith::constant(
//             self.context,
//             constant.into(),
//             self.location,
//         ));
//         let result_type = item.to_type(self.context);
//         match item.is_vectorized() {
//             true => self.append_operation_with_result(vector::broadcast(
//                 self.context,
//                 result_type,
//                 constant,
//                 self.location,
//             )),
//             false => constant,
//         }
//     }

//     pub fn create_int_constant_from_item(&self, item: ir::Type, constant: i64) -> Value<'a, 'a> {
//         let integer = item.storage_type().to_type(self.context);
//         let constant = IntegerAttribute::new(integer, constant);
//         let constant = self.append_operation_with_result(arith::constant(
//             self.context,
//             constant.into(),
//             self.location,
//         ));
//         let result_type = item.to_type(self.context);
//         match item.is_vectorized() {
//             true => self.append_operation_with_result(vector::broadcast(
//                 self.context,
//                 result_type,
//                 constant,
//                 self.location,
//             )),
//             false => constant,
//         }
//     }

//     pub fn create_constant_index(&self, constant: i64) -> Value<'a, 'a> {
//         let ty = Type::index(self.context);
//         let constant = IntegerAttribute::new(ty, constant);
//         self.append_operation_with_result(arith::constant(
//             self.context,
//             constant.into(),
//             self.location,
//         ))
//     }

//     pub fn cast_to_bool(&self, value: Value<'a, 'a>, item: ir::Type) -> Value<'a, 'a> {
//         let mut bool = IntegerType::new(self.context, 1).into();
//         if item.is_vectorized() {
//             bool = Type::vector(&[item.vector_size() as u64], bool);
//         }
//         self.append_operation_with_result(arith::trunci(value, bool, self.location))
//     }

//     pub fn cast_to_u8(&self, value: Value<'a, 'a>, item: ir::Type) -> Value<'a, 'a> {
//         let mut byte = IntegerType::new(self.context, 8).into();
//         if item.is_vectorized() {
//             byte = Type::vector(&[item.vector_size() as u64], byte);
//         }
//         self.append_operation_with_result(arith::extui(value, byte, self.location))
//     }
// }
