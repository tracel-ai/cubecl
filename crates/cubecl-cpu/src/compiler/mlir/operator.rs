use cubecl_core::ir::Operator;
use melior::{
    dialect::ods::vector,
    ir::{BlockLike, Value},
};

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&mut self, operator: &Operator, out: Value<'_, '_>) {
        match operator {
            Operator::Index(index_operator) => {
                let vector_type = self.item_to_type(index_operator.list.item);
                let operation = vector::load(
                    self.context,
                    vector_type,
                    self.get_variable(index_operator.list),
                    &[self.get_variable(index_operator.index)],
                    self.location,
                )
                .into();
                let load_value = self
                    .block()
                    .append_operation(operation)
                    .result(0)
                    .unwrap()
                    .into();
                self.block().append_operation(
                    vector::store(self.context, load_value, out, &[], self.location).into(),
                );
            }
            _ => todo!("{} is not yet implemented", operator),
        }
    }
}
