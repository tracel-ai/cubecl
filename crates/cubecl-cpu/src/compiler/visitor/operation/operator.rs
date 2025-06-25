use cubecl_core::ir::{Elem, IndexAssignOperator, IndexOperator, Operator};
use tracel_llvm::melior::{
    dialect::{
        arith::{self, CmpiPredicate},
        memref,
        ods::vector,
        scf,
    },
    ir::{Block, Region, attribute::IntegerAttribute},
};

use crate::compiler::visitor::prelude::*;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Variable) {
        match operator {
            Operator::Index(index) => {
                let index_value = self.get_index(index.index, out.item);
                let buffer_value = self.get_buffer_size(index.list);
                let condition = self.append_operation_with_result(arith::cmpi(
                    self.context,
                    CmpiPredicate::Ult,
                    index_value,
                    buffer_value,
                    self.location,
                ));
                let zero_attribute = IntegerAttribute::new(Type::index(self.context), 0).into();
                let zero = self.append_operation_with_result(arith::constant(
                    self.context,
                    zero_attribute,
                    self.location,
                ));
                let index_value = self.append_operation_with_result(arith::select(
                    condition,
                    index_value,
                    zero,
                    self.location,
                ));
                self.visit_index(index, index_value, out);
            }
            Operator::UncheckedIndex(index) => {
                let index_value = self.get_index(index.index, out.item);
                self.visit_index(index, index_value, out);
            }
            Operator::IndexAssign(index_assign) => {
                let index_assign_value =
                    self.get_index(index_assign.index, index_assign.value.item);
                let buffer_value = self.get_buffer_size(out);
                let condition = self.append_operation_with_result(arith::cmpi(
                    self.context,
                    CmpiPredicate::Ult,
                    index_assign_value,
                    buffer_value,
                    self.location,
                ));
                self.block().append_operation(scf::r#if(
                    condition,
                    &[],
                    {
                        let region = Region::new();
                        let block = Block::new(&[]);
                        let block_ref = region.append_block(block);
                        let previous = self.current_block;
                        self.current_block = block_ref;
                        self.visit_index_assign(index_assign, index_assign_value, out);
                        self.block()
                            .append_operation(scf::r#yield(&[], self.location));
                        self.current_block = previous;
                        region
                    },
                    Region::new(),
                    self.location,
                ));
            }
            Operator::UncheckedIndexAssign(index_assign) => {
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

    fn visit_index(&mut self, index: &IndexOperator, index_value: Value<'a, 'a>, out: Variable) {
        let memref = self.get_memory(index.list);
        let vector_type = index.list.item.to_type(self.context);
        let load_ssa = if out.item.is_vectorized() {
            self.append_operation_with_result(vector::load(
                self.context,
                vector_type,
                memref,
                &[index_value],
                self.location,
            ))
        } else {
            self.append_operation_with_result(memref::load(memref, &[index_value], self.location))
        };
        self.insert_variable(out, load_ssa);
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
