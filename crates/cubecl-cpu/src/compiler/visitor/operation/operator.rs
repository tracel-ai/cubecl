use cubecl_core::ir::{Elem, IndexAssignOperator, IndexOperator, Operator};
use tracel_llvm::melior::{
    dialect::{arith, memref, ods::vector},
    ir::attribute::DenseI64ArrayAttribute,
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Variable) {
        match operator {
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
            Operator::Cast(cast) => {
                self.visit_cast(cast.input, out);
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
            _ => todo!("{:?} is not yet implemented", operator),
        }
    }

    fn visit_index(
        &mut self,
        index: &IndexOperator,
        index_value: Value<'a, 'a>,
        out: Variable,
    ) -> Value<'a, 'a> {
        let vector_type = index.list.item.to_type(self.context);
        let value = if !self.is_memory(index.list) {
            let to_extract = self.get_variable(index.list);
            let zero = DenseI64ArrayAttribute::new(self.context, &[0]).into();
            self.append_operation_with_result(vector::extract(
                self.context,
                to_extract,
                &[index_value],
                zero,
                self.location,
            ))
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
        };
        value
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
        self.block().append_operation(operation);
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

        let value = if to_cast.elem().is_int() && out.elem().is_float() {
            self.append_operation_with_result(arith::sitofp(value, target, self.location))
        } else if to_cast.elem().is_float() && out.elem().is_int() {
            self.append_operation_with_result(arith::fptosi(value, target, self.location))
        } else if matches!(to_cast.elem(), Elem::Bool) || out.elem().is_int() {
            self.append_operation_with_result(arith::extui(value, target, self.location))
        } else if to_cast.elem() == out.elem() {
            value
        } else {
            panic!("Cast not implemented {} -> {}", to_cast.item, out.item);
        };

        self.insert_variable(out, value);
    }
}
