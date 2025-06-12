pub use super::Visitor;
use tracel_llvm::melior::{Context, ir::Type};

pub trait IntoType {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a>;
}
