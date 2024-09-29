use rspirv::spirv::Word;

use crate::{item::Item, variable::Variable};

#[derive(Clone, Debug)]
pub struct Slice {
    pub ptr: Variable,
    pub offset: Word,
    pub len: Word,
    pub const_len: Option<u32>,
    pub item: Item,
}

impl From<&Slice> for Variable {
    fn from(value: &Slice) -> Self {
        Variable::Slice {
            ptr: Box::new(value.ptr.clone()),
            offset: value.offset,
            len: value.len,
            const_len: value.const_len,
            item: value.item.clone(),
        }
    }
}
