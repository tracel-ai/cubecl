pub use super::Visitor;
use melior::{Context, ir::Type};

pub trait IntoType {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a>;
}
