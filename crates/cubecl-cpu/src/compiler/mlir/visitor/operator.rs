use cubecl_core::ir::{Elem, Operator, Variable};
use melior::{
    dialect::{arith, memref, ods::vector},
    ir::{BlockLike, Type},
};

use super::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Variable) {
        match operator {
            Operator::Index(index) | Operator::UncheckedIndex(index) => {
                let memref = self.get_memory(index.list);
                let vector_type = self.item_to_type(index.list.item);
                let index = self.get_index(index.index, index.list.item);
                let load_ssa = if out.item.vectorization.is_none() {
                    self.append_operation_with_result(memref::load(memref, &[index], self.location))
                } else {
                    self.append_operation_with_result(vector::load(
                        self.context,
                        vector_type,
                        memref,
                        &[index],
                        self.location,
                    ))
                };
                self.insert_variable(out, load_ssa);
            }
            Operator::IndexAssign(index_assign) | Operator::UncheckedIndexAssign(index_assign) => {
                let value = self.get_variable(index_assign.value);
                let index = self.get_index(index_assign.index, index_assign.value.item);
                let memref = self.get_memory(out);
                let operation = if index_assign.value.item.vectorization.is_none() {
                    memref::store(value, memref, &[index], self.location)
                } else {
                    vector::store(self.context, value, memref, &[index], self.location).into()
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
        let mut value = self.get_variable(to_cast);
        let target = self.item_to_type(out.item);

        if to_cast.item.vectorization.is_none() && out.item.vectorization.is_some() {
            let r#type = self.elem_to_type(to_cast.elem());
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
