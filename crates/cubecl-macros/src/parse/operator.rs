use syn::{BinOp, UnOp};

use crate::operator::Operator;

pub fn parse_binop(op: &BinOp) -> syn::Result<Operator> {
    let op = match op {
        BinOp::Add(_) => Operator::Add,
        BinOp::Sub(_) => Operator::Sub,
        BinOp::Mul(_) => Operator::Mul,
        BinOp::Div(_) => Operator::Div,
        BinOp::Rem(_) => Operator::Rem,
        BinOp::And(_) => Operator::And,
        BinOp::Or(_) => Operator::Or,
        BinOp::BitXor(_) => Operator::BitXor,
        BinOp::BitAnd(_) => Operator::BitAnd,
        BinOp::BitOr(_) => Operator::BitOr,
        BinOp::Shl(_) => Operator::Shl,
        BinOp::Shr(_) => Operator::Shr,
        BinOp::Eq(_) => Operator::Eq,
        BinOp::Lt(_) => Operator::Lt,
        BinOp::Le(_) => Operator::Le,
        BinOp::Ne(_) => Operator::Ne,
        BinOp::Ge(_) => Operator::Ge,
        BinOp::Gt(_) => Operator::Gt,
        BinOp::AddAssign(_) => Operator::AddAssign,
        BinOp::SubAssign(_) => Operator::SubAssign,
        BinOp::MulAssign(_) => Operator::MulAssign,
        BinOp::DivAssign(_) => Operator::DivAssign,
        BinOp::RemAssign(_) => Operator::RemAssign,
        BinOp::BitXorAssign(_) => Operator::BitXorAssign,
        BinOp::BitAndAssign(_) => Operator::BitAndAssign,
        BinOp::BitOrAssign(_) => Operator::BitOrAssign,
        BinOp::ShlAssign(_) => Operator::ShlAssign,
        BinOp::ShrAssign(_) => Operator::ShrAssign,
        op => Err(syn::Error::new_spanned(op, "Unsupported operator"))?,
    };
    Ok(op)
}

pub fn parse_unop(op: &UnOp) -> syn::Result<Operator> {
    let op = match op {
        UnOp::Deref(_) => Operator::Deref,
        UnOp::Not(_) => Operator::Not,
        UnOp::Neg(_) => Operator::Neg,
        op => Err(syn::Error::new_spanned(op, "Unsupported operator"))?,
    };
    Ok(op)
}
