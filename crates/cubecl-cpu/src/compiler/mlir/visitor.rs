use std::collections::HashMap;

use melior::{
    Context,
    ir::{Block, BlockLike, Location, Value},
};

pub struct Visitor<'a> {
    pub block: &'a Block<'a>,
    pub context: &'a Context,
    pub location: Location<'a>,
    pub current_local_variables: HashMap<u32, Value<'a, 'a>>,
    pub global_buffers: Vec<Value<'a, 'a>>,
}

impl<'a> Visitor<'a> {
    pub fn new(block: &'a Block<'a>, context: &'a Context, location: Location<'a>) -> Self {
        let current_local_variables = HashMap::new();
        let argument_count = block.argument_count();
        let mut global_buffers = Vec::with_capacity(argument_count);
        for i in 0..argument_count {
            global_buffers.push(block.argument(i).unwrap().into());
        }
        Self {
            block,
            context,
            location,
            current_local_variables,
            global_buffers,
        }
    }
}
