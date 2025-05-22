use cubecl_core::ir::Operator;
use melior::ir::{Block, Location, Value};

use super::prelude::*;

pub(super) trait VisitOperator {
    fn visit_with_out<'a, 'b, 'c>(
        &self,
        block: &Block,
        context: &'a Context,
        location: Location,
        out: Value<'b, 'c>,
    );
}

impl VisitOperator for Operator {
    fn visit_with_out<'a, 'b, 'c>(
        &self,
        block: &Block,
        context: &'a Context,
        location: Location,
        out: Value<'b, 'c>,
    ) {
        match self {
            Operator::Index(index_operator) => {
                // todo!("Do a vector load");
            }
            _ => todo!("{} is not yet implemented", self),
        }
    }
}
