pub use super::Visitor;
pub use cubecl_core::ir::Variable;
pub use cubecl_opt::Optimizer;
pub use tracel_llvm::melior::{
    Context, Error,
    helpers::{ArithBlockExt, BuiltinBlockExt},
    ir::{BlockLike, RegionLike, Type, TypeLike, Value, ValueLike, operation::OperationLike},
};

pub trait IntoType {
    fn to_type<'a>(self, context: &'a Context) -> Type<'a>;
    fn is_vectorized(&self) -> bool
    where
        Self: Sized,
    {
        false
    }
}
