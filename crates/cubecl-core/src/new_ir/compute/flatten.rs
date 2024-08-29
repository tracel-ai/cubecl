use std::{iter, num::NonZero, ops::DerefMut};

use cubecl_common::operator::Operator;

use crate::{
    ir::{
        self, BinaryOperator, Branch, ConditionalAssign, Elem, If, IfElse, Item, Loop, Metadata,
        RangeLoop, UnaryOperator, Variable,
    },
    new_ir::{Block, Expr, Expression, Statement, TensorExpression},
    prelude::{CubeContext, ExpandElement},
};

pub fn flatten_expr(expr: Expression, context: &mut CubeContext) -> Option<ExpandElement> {
    let res = match expr {
        Expression::Binary {
            left,
            operator,
            right,
            ty,
            vectorization,
        } => {
            let left = flatten_expr(*left, context).unwrap();
            let right = flatten_expr(*right, context).unwrap();
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
            let input = flatten_expr(*input, context).unwrap();
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
        Expression::Assigment { left, right, .. } => {
            let right = flatten_expr(*right, context).unwrap();
            match *left {
                Expression::Tensor(TensorExpression::Index { tensor, index }) => {
                    let index = flatten_expr(*index, context).unwrap();
                    let tensor = flatten_expr(*tensor, context).unwrap();
                    context.register(ir::Operator::IndexAssign(BinaryOperator {
                        lhs: *index,
                        rhs: *right,
                        out: *tensor,
                    }));
                    tensor
                }
                left => {
                    let left = flatten_expr(left, context).unwrap();
                    context.register(ir::Operator::Assign(UnaryOperator {
                        input: *right,
                        out: *left,
                    }));
                    left
                }
            }
        }
        Expression::Init { left, right, .. } => {
            let var = match *left {
                Expression::Variable { name, .. } => name,
                _ => unreachable!("Init only accepts variables for left"),
            };
            let right = flatten_expr(*right, context).unwrap();
            context.register_local(var, right.clone());
            right
        }
        Expression::Block(block) => flatten_block(block, &mut context.child())?,
        Expression::Break => {
            context.register(Branch::Break);
            None?
        }
        Expression::Cast {
            from,
            vectorization,
            to,
        } => {
            unimplemented!("Cast not yet implemented")
        }
        Expression::Continue => {
            unimplemented!("Continue not yet implemented")
        }
        Expression::ForLoop {
            range,
            unroll,
            variable,
            block,
        } => {
            let start = flatten_expr(*range.start, context).unwrap();
            let end = flatten_expr(*range.end, context).unwrap();
            let step = range.step.and_then(|expr| flatten_expr(*expr, context));
            let i = flatten_expr(*variable, context).unwrap();
            let mut scope = context.child();
            flatten_block(block, &mut scope);

            context.register(Branch::RangeLoop(RangeLoop {
                i: *i,
                start: *start,
                end: *end,
                step: step.map(Into::into),
                scope: scope.into_scope(),
            }));
            None?
        }
        Expression::WhileLoop {
            condition,
            mut block,
        } => {
            let break_cond = Expression::If {
                condition: Box::new(Expression::Unary {
                    input: condition,
                    operator: Operator::Not,
                    vectorization: None,
                    ty: Elem::Bool,
                }),
                then_block: Block {
                    inner: vec![Statement::Expression(Expression::Break)],
                    ret: Box::new(().expression_untyped()),
                    vectorization: None,
                    ty: Elem::Unit,
                },
                else_branch: None,
            };
            block.inner = iter::once(Statement::Expression(break_cond))
                .chain(block.inner)
                .collect();
            let mut scope = context.child();
            flatten_block(block, &mut scope);

            context.register(Branch::Loop(Loop {
                scope: scope.into_scope(),
            }));
            None?
        }
        Expression::Loop { block } => {
            let mut scope = context.child();
            flatten_block(block, &mut scope);

            context.register(Branch::Loop(Loop {
                scope: scope.into_scope(),
            }));
            None?
        }
        Expression::If {
            condition,
            then_block,
            else_branch,
        } => {
            let ty = then_block.ty;
            let has_ret = then_block.ret.ir_type() != Elem::Unit;
            let condition = flatten_expr(*condition, context).unwrap();

            if has_ret {
                let left = flatten_block(then_block, context).unwrap();
                let right = else_branch
                    .and_then(|expr| flatten_expr(*expr, context))
                    .unwrap();
                let out = context.create_local(Item::new(ty));
                ConditionalAssign::expand(
                    ConditionalAssign {
                        cond: *condition,
                        lhs: *left,
                        rhs: *right,
                        out: *out,
                    },
                    context.scope.borrow_mut().deref_mut(),
                );
                out
            } else if let Some(right) = else_branch {
                let mut scope_if = context.child();
                flatten_block(then_block, &mut scope_if).unwrap();
                let mut scope_else = context.child();
                flatten_expr(*right, &mut scope_else);
                context.register(Branch::IfElse(IfElse {
                    cond: *condition,
                    scope_if: scope_if.into_scope(),
                    scope_else: scope_else.into_scope(),
                }));
                None?
            } else {
                let mut scope = context.child();
                flatten_block(then_block, &mut scope);
                context.register(Branch::If(If {
                    cond: *condition,
                    scope: scope.into_scope(),
                }));
                None?
            }
        }
        Expression::Return { .. } => {
            context.register(Branch::Return);
            None?
        }
        Expression::Tensor(expr) => flatten_tensor_expr(expr, context)?,
        Expression::ArrayInit { size, init } => {
            let size = flatten_expr(*size, context).unwrap();
            // TODO: Init value, this isn't currently supported in the backend
            //let init = flatten_expr(*init, context).unwrap();
            let item = if let Some(vectorization) = init.vectorization() {
                Item::vectorized(init.ir_type(), vectorization.get())
            } else {
                Item::new(init.ir_type())
            };
            // I've already checked this is const in the macro
            let size = size.as_const().unwrap().as_u32();
            context.create_local_array(item, size)
        }
        Expression::KernelVar { kind, .. } => ExpandElement::Plain(kind),
        Expression::__Range(_) => unimplemented!("Range expressions don't exist post expansion"),
    };
    Some(res)
}

pub fn flatten_statement(stmt: Statement, context: &mut CubeContext) -> Option<ExpandElement> {
    match stmt {
        Statement::Local { variable, .. } => flatten_expr(variable, context),
        Statement::Expression(expr) => flatten_expr(expr, context),
    }
}

pub fn flatten_block(block: Block, scope: &mut CubeContext) -> Option<ExpandElement> {
    for inner in block.inner {
        flatten_statement(inner, scope);
    }
    flatten_expr(*block.ret, scope)
}

fn flatten_tensor_expr(expr: TensorExpression, context: &mut CubeContext) -> Option<ExpandElement> {
    let res = match expr {
        TensorExpression::Stride { tensor, dim } => {
            let tensor = flatten_expr(*tensor, context).unwrap();
            let dim = flatten_expr(*dim, context).unwrap();
            let out = context.create_local(Item::new(Elem::UInt));
            context.register(Metadata::Stride {
                dim: *dim,
                var: *tensor,
                out: out.clone().into(),
            });
            out
        }
        TensorExpression::Shape { tensor, dim } => {
            let tensor = flatten_expr(*tensor, context).unwrap();
            let dim = flatten_expr(*dim, context).unwrap();
            let out = context.create_local(Item::new(Elem::UInt));
            context.register(Metadata::Shape {
                dim: *dim,
                var: *tensor,
                out: out.clone().into(),
            });
            out
        }
        TensorExpression::Length { tensor } => {
            let tensor = flatten_expr(*tensor, context).unwrap();
            let out = context.create_local(Item::new(Elem::UInt));
            context.register(Metadata::Length {
                var: *tensor,
                out: out.clone().into(),
            });
            out
        }
        TensorExpression::Rank { .. } => ExpandElement::Plain(Variable::Rank),
        TensorExpression::Index { tensor, index } => {
            let tensor = flatten_expr(*tensor, context).unwrap();
            let index = flatten_expr(*index, context).unwrap();
            let out = context.create_local(tensor.item());
            context.register(ir::Operator::Index(BinaryOperator {
                rhs: *index,
                lhs: *tensor,
                out: out.clone().into(),
            }));
            out
        }
        TensorExpression::Slice { ranges, tensor } => {
            let input = flatten_expr(*tensor.clone(), context).unwrap();
            assert_eq!(ranges.len(), 1, "Multi-slices not currently supported");
            let start = flatten_expr(*ranges[0].start.clone(), context).unwrap();
            let end = ranges[0]
                .end
                .clone()
                .and_then(|expr| flatten_expr(*expr, context))
                .unwrap_or_else(|| {
                    flatten_tensor_expr(TensorExpression::Length { tensor }, context).unwrap()
                });
            let out = context.create_slice(input.item());

            context.register(ir::Operator::Slice(ir::SliceOperator {
                input: *input,
                start: *start,
                end: *end,
                out: *out,
            }));

            out
        }
        TensorExpression::__SliceRange(_) => unimplemented!("Slice ranges don't exist at runtime"),
    };
    Some(res)
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
