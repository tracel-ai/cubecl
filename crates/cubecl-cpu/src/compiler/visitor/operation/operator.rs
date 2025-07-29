use cubecl_core::ir::{Elem, IndexAssignOperator, IndexOperator, Operator};
use tracel_llvm::melior::{
    dialect::{
        arith, memref,
        ods::{self, vector},
    },
    ir::{Operation, attribute::DenseI64ArrayAttribute},
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Variable) {
        match operator {
            Operator::And(and) => {
                let lhs = self.get_variable(and.lhs);
                let rhs = self.get_variable(and.rhs);
                let value = self.append_operation_with_result(arith::andi(lhs, rhs, self.location));
                self.insert_variable(out, value);
            }
            Operator::Cast(cast) => {
                self.visit_cast(cast.input, out);
            }
            Operator::CopyMemory(copy_memory) => {
                let memref = self.get_memory(copy_memory.input);
                let in_index = self.get_index(copy_memory.in_index, copy_memory.input.item);
                let value = self.append_operation_with_result(memref::load(
                    memref,
                    &[in_index],
                    self.location,
                ));
                let out_memref = self.get_memory(out);
                let out_index = self.get_index(copy_memory.out_index, out.item);
                self.block.append_operation(memref::store(
                    value,
                    out_memref,
                    &[out_index],
                    self.location,
                ));
            }
            Operator::CopyMemoryBulk(_copy_memory_bulk) => {
                todo!("copy_memory_bulk is not implemented {}", operator)
            }
            Operator::Index(index) | Operator::UncheckedIndex(index) => {
                let index_value = self.get_index(index.index, out.item);
                let load_ssa = self.visit_index(index, index_value, out);
                self.insert_variable(out, load_ssa);
            }
            Operator::IndexAssign(index_assign) | Operator::UncheckedIndexAssign(index_assign) => {
                let index_assign_value =
                    self.get_index(index_assign.index, index_assign.value.item);
                self.visit_index_assign(index_assign, index_assign_value, out)
            }
            Operator::InitLine(init_line) => {
                let inputs: Vec<_> = init_line
                    .inputs
                    .iter()
                    .map(|input| self.get_variable(*input))
                    .collect();
                let result = out.item.to_type(self.context);
                let init_line = self.append_operation_with_result(vector::from_elements(
                    self.context,
                    result,
                    &inputs,
                    self.location,
                ));
                self.insert_variable(out, init_line);
            }
            Operator::Not(not) => {
                let lhs = self.get_variable(not.input);
                let mask = self.create_int_constant_from_item(not.input.item, -1);
                let value =
                    self.append_operation_with_result(arith::xori(lhs, mask, self.location));
                self.insert_variable(out, value);
            }
            Operator::Or(or) => {
                let lhs = self.get_variable(or.lhs);
                let rhs = self.get_variable(or.rhs);
                let value = self.append_operation_with_result(arith::ori(lhs, rhs, self.location));
                self.insert_variable(out, value);
            }
            Operator::Reinterpret(reinterpret) => {
                let target_type = out.item.to_type(self.context);
                let input = self.get_variable(reinterpret.input);
                let value = self.append_operation_with_result(arith::bitcast(
                    input,
                    target_type,
                    self.location,
                ));
                self.insert_variable(out, value);
            }
            Operator::Select(select) => {
                let condition = self.get_variable(select.cond);
                let (then, or_else) = self.get_binary_op_variable(select.then, select.or_else);
                let value = self.append_operation_with_result(arith::select(
                    condition,
                    then,
                    or_else,
                    self.location,
                ));
                self.insert_variable(out, value);
            }
        }
    }

    fn visit_index(
        &mut self,
        index: &IndexOperator,
        index_value: Value<'a, 'a>,
        out: Variable,
    ) -> Value<'a, 'a> {
        let vector_type = index.list.item.to_type(self.context);
        if !self.is_memory(index.list) {
            let to_extract = self.get_variable(index.list);
            let zero =
                DenseI64ArrayAttribute::new(self.context, &[Visitor::into_i64(index.index)]).into();
            // Extract operation on vector with dynamic indexes is badly supported by MLIR
            let vector_extract =
                vector::extract(self.context, to_extract, &[], zero, self.location);
            self.append_operation_with_result(vector_extract)
        } else if out.item.is_vectorized() {
            let memref = self.get_memory(index.list);
            self.append_operation_with_result(vector::load(
                self.context,
                vector_type,
                memref,
                &[index_value],
                self.location,
            ))
        } else {
            let memref = self.get_memory(index.list);
            self.append_operation_with_result(memref::load(memref, &[index_value], self.location))
        }
    }

    fn visit_index_assign(
        &mut self,
        index_assign: &IndexAssignOperator,
        index_assign_value: Value<'a, 'a>,
        out: Variable,
    ) {
        let value = self.get_variable(index_assign.value);
        let memref = self.get_memory(out);
        let operation = if index_assign.value.item.is_vectorized() {
            vector::store(
                self.context,
                value,
                memref,
                &[index_assign_value],
                self.location,
            )
            .into()
        } else {
            memref::store(value, memref, &[index_assign_value], self.location)
        };
        self.block.append_operation(operation);
    }

    fn visit_cast(&mut self, to_cast: Variable, out: Variable) {
        let mut value = self.get_variable(to_cast);
        let target = out.item.to_type(self.context);

        if !to_cast.item.is_vectorized() && out.item.is_vectorized() {
            let r#type = to_cast.elem().to_type(self.context);
            let vector_type = Type::vector(&[out.vectorization_factor() as u64], r#type);
            value = self.append_operation_with_result(vector::splat(
                self.context,
                vector_type,
                value,
                self.location,
            ));
        };

        let value = if to_cast.elem().is_int() == out.elem().is_int() {
            self.get_cast_same_type_category(to_cast.elem(), out.elem(), target, value)
        } else {
            self.get_cast_different_type_category(to_cast.elem(), out.elem(), target, value)
        };

        self.insert_variable(out, value);
    }

    fn get_cast_different_type_category(
        &self,
        to_cast: Elem,
        out: Elem,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Value<'a, 'a> {
        if to_cast.is_int() {
            self.append_operation_with_result(self.cast_int_to_float(to_cast, target, value))
        } else {
            self.append_operation_with_result(self.cast_float_to_int(out, target, value))
        }
    }

    fn cast_float_to_int(
        &self,
        out: Elem,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Operation<'a> {
        if out.is_signed_int() {
            arith::fptosi(value, target, self.location)
        } else {
            arith::fptoui(value, target, self.location)
        }
    }

    fn cast_int_to_float(
        &self,
        to_cast: Elem,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Operation<'a> {
        if to_cast.is_signed_int() {
            arith::sitofp(value, target, self.location)
        } else {
            arith::uitofp(value, target, self.location)
        }
    }

    fn get_cast_same_type_category(
        &self,
        to_cast: Elem,
        out: Elem,
        target: Type<'a>,
        value: Value<'a, 'a>,
    ) -> Value<'a, 'a> {
        if to_cast.size() > out.size() {
            self.append_operation_with_result(self.get_trunc(to_cast, target, value))
        } else if to_cast.size() < out.size() {
            self.append_operation_with_result(self.get_ext(to_cast, target, value))
        } else {
            value
        }
    }

    fn get_trunc(&self, to_cast: Elem, target: Type<'a>, value: Value<'a, 'a>) -> Operation<'a> {
        if to_cast.is_int() {
            arith::trunci(value, target, self.location)
        } else {
            ods::arith::truncf(self.context, target, value, self.location).into()
        }
    }

    fn get_ext(&self, to_cast: Elem, target: Type<'a>, value: Value<'a, 'a>) -> Operation<'a> {
        if to_cast.is_signed_int() {
            arith::extsi(value, target, self.location)
        } else if to_cast.is_unsigned_int() {
            arith::extui(value, target, self.location)
        } else {
            arith::extf(value, target, self.location)
        }
    }
}
