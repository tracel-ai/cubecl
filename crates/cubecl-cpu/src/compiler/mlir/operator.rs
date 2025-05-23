use cubecl_core::ir::Operator;
use melior::ir::Value;

use super::visitor::Visitor;

impl<'a> Visitor<'a> {
    pub fn visit_operator_with_out(&self, operator: &Operator, _out: Value<'a, 'a>) {
        match operator {
            Operator::Index(_index_operator) => {
                // let vector_type = self.item_to_type(index_operator.list.item);
                // self.block.append_operation(
                //     *vector::load(self.context, vector_type, base, indices, location)
                //         .as_operation(),
                // );
            }
            _ => todo!("{} is not yet implemented", operator),
        }
    }
}
