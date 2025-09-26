#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FlashIdent {
    Query,
    Key,
    Softmax,
    Value,
    Mask,
    Out,
}
