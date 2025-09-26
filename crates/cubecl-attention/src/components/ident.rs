#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum AttentionIdent {
    Query,
    Key,
    Softmax,
    Value,
    Mask,
    Out,
}
