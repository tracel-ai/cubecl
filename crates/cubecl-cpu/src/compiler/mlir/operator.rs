use cubecl_core::ir::Operator;
use melior::{
    dialect::memref,
    ir::{BlockLike, Value},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Value<'_, '_>) {
        match operator {
            Operator::Index(index_operator) => {
                let memref = self.get_variable(index_operator.list);
                let operation = memref::load(memref, &[], self.location).into();
                let load_ssa = self
                    .block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into();
                let operation = memref::store(load_ssa, out, &[], self.location).into();
                self.block().append_operation(operation);
            }
            _ => todo!("{} is not yet implemented", operator),
        }
    }
}
