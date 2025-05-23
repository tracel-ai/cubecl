use std::collections::HashMap;

use cubecl_core::ir::Variable;
use melior::{
    Context,
    ir::{Block, Location, Value},
};

pub struct Visitor<'a> {
    pub block: &'a Block<'a>,
    pub context: &'a Context,
    pub location: Location<'a>,
    pub current_variables: HashMap<Variable, Value<'a, 'a>>,
}

impl<'a> Visitor<'a> {
    pub fn new(current_block: &'a Block<'a>, context: &'a Context, location: Location<'a>) -> Self {
        let current_variables = HashMap::new();
        Self {
            block: current_block,
            context,
            location,
            current_variables,
        }
    }
}
