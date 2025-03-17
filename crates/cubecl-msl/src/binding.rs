use crate::{AddressSpace, Item};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Binding {
    pub address_space: AddressSpace,
    pub item: Item,
    pub size: Option<usize>,
}
