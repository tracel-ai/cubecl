#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Ident {
    Query,
    Key,
    Value,
    Mask,
    Out,
}
