#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum FlashIdent {
    Query,
    Key,
    ScoreProb,
    Value,
    Mask,
    Out,
}
