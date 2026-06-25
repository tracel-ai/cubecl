//! Convert unsupported types ahead of time. Also convert auto-promoted types manually so we can
//! properly preserve the semantics of the actual code in IR.

macro_rules! no_half {
    ($ty: ty) => {
        // TODO
    };
}
pub(crate) use no_half;

macro_rules! promotes_int {
    ($ty: ty) => {
        // TODO
    };
}
pub(crate) use promotes_int;
