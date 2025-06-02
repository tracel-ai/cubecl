use cubecl_core::ir::{Elem, Operator, Variable};
use melior::{
    dialect::{arith, memref, ods::vector},
    ir::BlockLike,
};

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Variable) {
        match operator {
            Operator::Index(index) | Operator::UncheckedIndex(index) => {
                let memref = self.get_variable(index.list);
                let vector_type = self.item_to_type(index.list.item);
                let index = self.get_index(index.index, index.list.item);
                let operation =
                    vector::load(self.context, vector_type, memref, &[index], self.location).into();
                let load_ssa = self
                    .block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into();
                self.insert_variable(out, load_ssa);
            }
            Operator::IndexAssign(index_assign) | Operator::UncheckedIndexAssign(index_assign) => {
                let memref = self.get_variable(index_assign.value);
                let index = self.get_index(index_assign.index, index_assign.value.item);
                let out_value = self.get_variable(out);
                let operation = if index_assign.value.item.vectorization.is_none() {
                    memref::store(memref, out_value, &[index], self.location)
                } else {
                    vector::store(self.context, memref, out_value, &[index], self.location).into()
                };
                self.block().append_operation(operation);
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

    pub fn visit_cast(&mut self, to_cast: Variable, out: Variable) {
        let value = self.get_variable(to_cast);
        let target = self.item_to_type(out.item);
        let value = if to_cast.elem().is_int() && self.is_float(out.elem()) {
            self.append_operation_with_result(arith::sitofp(value, target, self.location))
        } else if self.is_float(to_cast.elem()) && out.elem().is_int() {
            self.append_operation_with_result(arith::fptosi(value, target, self.location))
        } else if matches!(to_cast.elem(), Elem::Bool) || out.elem().is_int() {
            self.append_operation_with_result(arith::extui(value, target, self.location))
        } else {
            panic!("Cast not implemented {} -> {}", to_cast.elem(), out.elem());
        };
        self.insert_variable(out, value);
    }
}
