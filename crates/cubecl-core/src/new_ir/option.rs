use std::marker::PhantomData;

use super::{DynamicExpr, Expr, PartialExpand, StaticExpand, StaticExpanded};

impl<T: Expr<Output = T> + 'static> StaticExpand for Option<T> {
    type Expanded = OptionStatic<T>;
}

impl<T: Expr<Output = T> + 'static> PartialExpand for Option<T> {
    type Expanded = OptionExpand<T>;

    fn partial_expand(self) -> Self::Expanded {
        OptionExpand(self)
    }
}

pub struct OptionStatic<T: Expr<Output = T> + 'static>(PhantomData<T>);
pub struct OptionExpand<T: Expr<Output = T> + 'static>(Option<T>);

impl<T: Expr<Output = T> + 'static> StaticExpanded for OptionStatic<T> {
    type Unexpanded = Option<T>;
}

impl<T: Expr<Output = T> + 'static> StaticExpanded for OptionExpand<T> {
    type Unexpanded = Option<T>;
}

impl<T: Expr<Output = T> + 'static> OptionStatic<T> {
    pub fn unwrap_or<Other: Expr<Output = T> + 'static>(
        this: Option<T>,
        other: Other,
    ) -> DynamicExpr<T> {
        match this {
            Some(this) => DynamicExpr(Box::new(this)),
            None => DynamicExpr(Box::new(other)),
        }
    }

    pub fn unwrap_or_else<Other: Expr<Output = T> + 'static>(
        this: Option<T>,
        other: impl Fn() -> Other,
    ) -> DynamicExpr<T> {
        match this {
            Some(this) => DynamicExpr(Box::new(this)),
            None => DynamicExpr(Box::new(other())),
        }
    }
}

impl<T: Expr<Output = T> + 'static> OptionExpand<T> {
    pub fn unwrap_or<Other: Expr<Output = T> + 'static>(self, other: Other) -> DynamicExpr<T> {
        match self.0 {
            Some(this) => DynamicExpr(Box::new(this)),
            None => DynamicExpr(Box::new(other)),
        }
    }

    pub fn unwrap_or_else<Other: Expr<Output = T> + 'static>(
        self,
        other: impl Fn() -> Other,
    ) -> DynamicExpr<T> {
        match self.0 {
            Some(this) => DynamicExpr(Box::new(this)),
            None => DynamicExpr(Box::new(other())),
        }
    }
}
