pub use super::Visitor;
pub use tracel_llvm::mlir_rs::{
    Context, Error,
    helpers::{ArithBlockExt, BuiltinBlockExt},
    ir::{BlockLike, RegionLike, Type, TypeLike, ValueLike, operation::OperationLike},
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
