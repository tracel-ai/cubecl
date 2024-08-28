use std::num::NonZero;

use cubecl_common::operator::Operator;

use crate::{
    ir::{self, BinaryOperator, Elem, Item, UnaryOperator, Variable},
    new_ir::{Expression, Statement},
    prelude::{CubeContext, ExpandElement},
};

pub fn flatten_expr(expr: Expression, context: &mut CubeContext) -> ExpandElement {
    match expr {
        Expression::Binary {
            left,
            operator,
            right,
            ty,
            vectorization,
        } => {
            let left = flatten_expr(*left, context);
            let right = flatten_expr(*right, context);
            let out = if operator.is_assign() {
                left.clone()
            } else {
                context.create_local(item(ty, vectorization))
            };
            let operation = map_bin_op(
                operator,
                BinaryOperator {
                    lhs: *left,
                    rhs: *right,
                    out: *out,
                },
            );
            context.register(operation);
            out
        }
        Expression::Unary {
            input,
            operator,
            vectorization,
            ty,
        } => {
            let input = flatten_expr(*input, context);
            let out = context.create_local(item(ty, vectorization));
            context.register(map_un_op(
                operator,
                UnaryOperator {
                    input: *input,
                    out: *out,
                },
            ));
            out
        }
        Expression::Variable {
            name,
            vectorization,
            ty,
        } => {
            if let Some(var) = context.get_local(&name) {
                var
            } else {
                // This must be a declaration, because non-existing variables don't compile
                let new = context.create_local(item(ty, vectorization));
                context.register_local(name, new.clone());
                new
            }
        }
        Expression::Global {
            index,
            global_ty,
            vectorization,
            ty,
        } => match global_ty {
            super::GlobalType::Scalar => context.scalar(index, ty),
            super::GlobalType::InputArray => context.input(index, item(ty, vectorization)),
            super::GlobalType::OutputArray => context.output(index, item(ty, vectorization)),
        },
        Expression::FieldAccess {
            base,
            name,
            vectorization,
            ty,
        } => todo!(),
        Expression::Literal { value, .. } => ExpandElement::Plain(Variable::ConstantScalar(value)),
        Expression::Assigment { left, right, .. } | Expression::Init { left, right, .. } => {
            let left = flatten_expr(*left, context);
            let right = flatten_expr(*right, context);
            context.register(ir::Operator::Assign(UnaryOperator {
                input: *right,
                out: *left,
            }));
            left
        }
        Expression::Block {
            inner,
            ret,
            vectorization,
            ty,
        } => todo!(),
        Expression::Break => todo!(),
        Expression::Cast {
            from,
            vectorization,
            to,
        } => todo!(),
        Expression::Continue => todo!(),
        Expression::ForLoop {
            range,
            unroll,
            variable,
            block,
        } => todo!(),
        Expression::WhileLoop { condition, block } => todo!(),
        Expression::Loop { block } => todo!(),
        Expression::If {
            condition,
            then_block,
            else_branch,
        } => todo!(),
        Expression::Return { expr } => todo!(),
        Expression::Tensor(_) => todo!(),
        Expression::__Range(_) => todo!(),
        Expression::ArrayInit { size, init } => todo!(),
    }
}

pub fn flatten_statement(stmt: Statement, context: &mut CubeContext) -> ExpandElement {
    match stmt {
        Statement::Local { variable, .. } => flatten_expr(variable, context),
        Statement::Expression(expr) => flatten_expr(expr, context),
    }
}

fn map_bin_op(operator: Operator, bin_op: BinaryOperator) -> ir::Operator {
    match operator {
        Operator::Add => ir::Operator::Add(bin_op),
        Operator::Sub => ir::Operator::Sub(bin_op),
        Operator::Mul => ir::Operator::Mul(bin_op),
        Operator::Div => ir::Operator::Div(bin_op),
        Operator::Rem => ir::Operator::Remainder(bin_op),
        Operator::AddAssign => ir::Operator::Add(bin_op),
        Operator::SubAssign => ir::Operator::Sub(bin_op),
        Operator::MulAssign => ir::Operator::Mul(bin_op),
        Operator::DivAssign => ir::Operator::Div(bin_op),
        Operator::RemAssign => ir::Operator::Remainder(bin_op),
        Operator::Eq => ir::Operator::Equal(bin_op),
        Operator::Ne => ir::Operator::NotEqual(bin_op),
        Operator::Lt => ir::Operator::Lower(bin_op),
        Operator::Le => ir::Operator::LowerEqual(bin_op),
        Operator::Ge => ir::Operator::GreaterEqual(bin_op),
        Operator::Gt => ir::Operator::Greater(bin_op),
        Operator::And => ir::Operator::And(bin_op),
        Operator::Or => ir::Operator::Or(bin_op),
        Operator::BitXor => ir::Operator::BitwiseXor(bin_op),
        Operator::BitAnd => ir::Operator::BitwiseAnd(bin_op),
        Operator::BitOr => ir::Operator::Or(bin_op),
        Operator::BitXorAssign => ir::Operator::BitwiseXor(bin_op),
        Operator::BitAndAssign => ir::Operator::BitwiseAnd(bin_op),
        Operator::BitOrAssign => ir::Operator::Or(bin_op),
        Operator::Shl => ir::Operator::ShiftLeft(bin_op),
        Operator::Shr => ir::Operator::ShiftRight(bin_op),
        Operator::ShlAssign => ir::Operator::ShiftLeft(bin_op),
        Operator::ShrAssign => ir::Operator::ShiftRight(bin_op),
        _ => unreachable!("Operator must be binary"),
    }
}

fn map_un_op(operator: Operator, un_op: UnaryOperator) -> ir::Operator {
    match operator {
        Operator::Deref => unimplemented!("Deref not yet supported"),
        Operator::Not => ir::Operator::Not(un_op),
        Operator::Neg => ir::Operator::Neg(un_op),
        _ => unreachable!("Operator must be unary"),
    }
}

fn item(ty: Elem, vectorization: Option<NonZero<u8>>) -> Item {
    vectorization
        .map(|vec| Item::vectorized(ty, vec.get()))
        .unwrap_or_else(|| Item::new(ty))
}
