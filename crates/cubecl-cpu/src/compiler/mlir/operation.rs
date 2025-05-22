use cubecl_core::ir::Operation;
use melior::ir::{Block, Location, Value};

use super::{operator::VisitOperator, prelude::*};

pub(super) trait VisitOperation {
    fn visit_with_out<'a, 'b, 'c>(
        &self,
        block: &Block,
        context: &'a Context,
        location: Location,
        out: Value<'b, 'c>,
    );
}

impl VisitOperation for Operation {
    fn visit_with_out<'a, 'b, 'c>(
        &self,
        block: &Block,
        context: &'a Context,
        location: Location,
        out: Value<'b, 'c>,
    ) {
        match self {
            Operation::Operator(operator) => {
                operator.visit_with_out(block, context, location, out);
            }
            _ => todo!("Operation are not all implemented yet."),
        }
    }
}
