use std::{iter, num::NonZero, ops::DerefMut};

use cubecl_common::operator::Operator;

use crate::{
    compute::GlobalType,
    ir::{
        self, BinaryOperator, Branch, ClampOperator, ConditionalAssign, Elem, FmaOperator, If,
        IfElse, InitOperator, Item, Loop, Metadata, Operation, RangeLoop, Subcube, UnaryOperator,
        Variable,
    },
    new_ir::{Block, Expr, Expression, Statement, SubcubeExpression, SubcubeOp, TensorExpression},
    prelude::{CubeContext, ExpandElement},
};

use super::Var;

impl Expression {
    pub fn flatten(self, context: &mut CubeContext) -> Option<ExpandElement> {
        let res = match self {
            Expression::Binary {
                left,
                operator,
                right,
                ty,
                vectorization,
            } => {
                if matches!(*left, Expression::Tensor(TensorExpression::Index { .. }))
                    && operator.is_assign()
                {
                    return split_assign_op(*left, *right, operator, context);
                }

                let left = left.flatten(context).unwrap();
                let right = right.flatten(context).unwrap().as_variable();
                if operator.is_assign() {
                    let bin_op = BinaryOperator {
                        lhs: left.as_variable(),
                        rhs: right,
                        out: left.as_variable(),
                    };
                    context.register(map_bin_op(operator, bin_op));
                    left
                } else {
                    let left = left.into_variable();
                    let out = context.create_local(item(ty, vectorization));
                    let bin_op = BinaryOperator {
                        lhs: left,
                        rhs: right,
                        out: out.as_variable(),
                    };
                    context.register(map_bin_op(operator, bin_op));
                    out
                }
            }
            Expression::Unary {
                input,
                operator,
                vectorization,
                ty,
            } => {
                let input = input.flatten(context).unwrap().into_variable();
                let out = context.create_local(item(ty, vectorization));
                context.register(map_un_op(
                    operator,
                    UnaryOperator {
                        input,
                        out: out.as_variable(),
                    },
                ));
                out
            }
            Expression::Variable(Var {
                name,
                vectorization,
                ty,
            }) => {
                if let Some(var) = context.get_local(&name) {
                    var
                } else {
                    // This must be a declaration, because non-existing variables don't compile
                    let new = context.create_local(item(ty, vectorization));
                    context.register_local(name, new.as_weak());
                    new
                }
            }
            Expression::Global {
                index,
                global_ty,
                vectorization,
                ty,
            } => match global_ty {
                GlobalType::Scalar => context.scalar(index, ty),
                GlobalType::InputArray => context.input(index, item(ty, vectorization)),
                GlobalType::OutputArray => context.output(index, item(ty, vectorization)),
            },
            Expression::FieldAccess { .. } => todo!("Field access"),
            Expression::Literal { value, .. } => {
                ExpandElement::Plain(Variable::ConstantScalar(value))
            }
            Expression::Assigment { left, right, .. } => {
                let right = right.flatten(context).unwrap().into_variable();
                match *left {
                    Expression::Tensor(TensorExpression::Index { tensor, index, .. }) => {
                        let index = index.flatten(context).unwrap().as_variable();
                        let tensor = tensor.flatten(context).unwrap();
                        context.register(ir::Operator::IndexAssign(BinaryOperator {
                            lhs: index,
                            rhs: right,
                            out: tensor.as_variable(),
                        }));
                        tensor
                    }
                    left => {
                        let left = left.flatten(context).unwrap();
                        context.register(ir::Operator::Assign(UnaryOperator {
                            input: right,
                            out: left.as_variable(),
                        }));
                        left
                    }
                }
            }
            Expression::Init { left, right, .. } => {
                let right = right.flatten(context).unwrap();
                context.register_local(left.name, right.as_weak());
                right
            }
            Expression::Block(block) => flatten_block(block, &mut context.child())?,
            Expression::Break => {
                context.register(Branch::Break);
                None?
            }
            Expression::Cast {
                from,
                to,
                vectorization,
            } => {
                let input = from.flatten(context).unwrap().into_variable();
                let out = context.create_local(item(to, vectorization));
                context.register(ir::Operator::Assign(UnaryOperator {
                    input,
                    out: out.as_variable(),
                }));
                out
            }
            Expression::BitCast {
                from,
                vectorization,
                to,
            } => {
                let input = from.flatten(context).unwrap().into_variable();
                let out = context.create_local(item(to, vectorization));
                context.register(ir::Operator::Bitcast(UnaryOperator {
                    input,
                    out: out.as_variable(),
                }));
                out
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
                if unroll {
                    let start = range.start.as_lit().unwrap().as_usize();
                    let end = range.end.as_lit().unwrap().as_usize();
                    let step = range.step.map(|it| it.as_lit().unwrap().as_usize());

                    let mut func = |i: usize| {
                        let value = ExpandElement::Plain(variable.ty.constant_from_u64(i as u64));
                        let mut scope = context.child();
                        scope.register_local(variable.name.clone(), value.as_weak());
                        flatten_block(block.clone(), &mut scope)
                    };

                    match (step, range.inclusive) {
                        (None, true) => {
                            for i in start..=end {
                                func(i);
                            }
                        }
                        (None, false) => {
                            for i in start..end {
                                func(i);
                            }
                        }
                        (Some(step), true) => {
                            for i in (start..=end).step_by(step) {
                                func(i);
                            }
                        }
                        (Some(step), false) => {
                            for i in (start..end).step_by(step) {
                                func(i);
                            }
                        }
                    }
                    None?
                } else {
                    let start = range.start.flatten(context).unwrap().as_variable();
                    let end = range.end.flatten(context).unwrap().as_variable();
                    let step = range.step.and_then(|expr| expr.flatten(context));
                    let mut scope = context.child();
                    let i = scope
                        .scope
                        .borrow_mut()
                        .create_local_undeclared(start.item());
                    let var = ExpandElement::Plain(i);
                    scope.register_local(variable.name, var.as_weak());
                    flatten_block(block, &mut scope);

                    context.register(Branch::RangeLoop(RangeLoop {
                        i,
                        start,
                        end,
                        step: step.as_ref().map(|it| it.as_variable()),
                        scope: scope.into_scope(),
                    }));
                    None?
                }
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
                let cond = condition.flatten(context).unwrap().as_variable();

                if has_ret {
                    let lhs = flatten_block(then_block, context).unwrap().into_variable();
                    let rhs = else_branch
                        .and_then(|expr| expr.flatten(context))
                        .unwrap()
                        .as_variable();
                    let out = context.create_local(Item::new(ty));
                    ConditionalAssign::expand(
                        ConditionalAssign {
                            cond,
                            lhs,
                            rhs,
                            out: out.as_variable(),
                        },
                        context.scope.borrow_mut().deref_mut(),
                    );
                    out
                } else if let Some(right) = else_branch {
                    let mut scope_if = context.child();
                    flatten_block(then_block, &mut scope_if).unwrap();
                    let mut scope_else = context.child();
                    match *right {
                        Expression::Block(block) => flatten_block(block, &mut scope_else),
                        right => right.flatten(&mut scope_else),
                    };
                    context.register(Branch::IfElse(IfElse {
                        cond,
                        scope_if: scope_if.into_scope(),
                        scope_else: scope_else.into_scope(),
                    }));
                    None?
                } else {
                    let mut scope = context.child();
                    flatten_block(then_block, &mut scope);
                    context.register(Branch::If(If {
                        cond,
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
            Expression::ArrayInit {
                size,
                ty,
                vectorization,
            } => context.create_local_array(item(ty, vectorization), size),
            Expression::KernelVar { kind, .. } => ExpandElement::Plain(kind),
            Expression::Subcube(subcube) => flatten_subcube(subcube, context)?,
            Expression::Cmma(cmma) => cmma.flatten(context)?,

            Expression::__Range(_) => {
                unimplemented!("Range expressions don't exist post expansion")
            }
            Expression::Clamp {
                input,
                min,
                max,
                vectorization,
                ty,
            } => {
                let input = input.flatten(context).unwrap().into_variable();
                let min = min.flatten(context).unwrap().as_variable();
                let max = max.flatten(context).unwrap().as_variable();
                let out = context.create_local(item(ty, vectorization));
                context.register(ir::Operator::Clamp(ClampOperator {
                    input,
                    min_value: min,
                    max_value: max,
                    out: out.as_variable(),
                }));
                out
            }
            Expression::Atomic(expr) => expr.flatten(context)?,
            Expression::SharedMemory(expr) => expr.flatten(context)?,
            Expression::Fma {
                a,
                b,
                c,
                ty,
                vectorization,
            } => {
                let a = a.flatten(context).unwrap().into_variable();
                let b = b.flatten(context).unwrap().as_variable();
                let c = c.flatten(context).unwrap().as_variable();
                let output = context.create_local(item(ty, vectorization));
                let out = output.as_variable();

                context.register(ir::Operator::Fma(FmaOperator { a, b, c, out }));

                output
            }
            Expression::RuntimeStruct { .. } => {
                todo!("RuntimeStruct")
            }
            Expression::Sync(sync) => {
                context.register(sync);
                None?
            }
            Expression::Once(once) => {
                once.get_or_expand_with(|expr| expr.flatten(context).unwrap())
            }
        };
        Some(res)
    }
}

pub fn flatten_statement(stmt: Statement, context: &mut CubeContext) -> Option<ExpandElement> {
    match stmt {
        Statement::Local { variable, .. } => variable.flatten(context),
        Statement::Expression(expr) => expr.flatten(context),
    }
}

pub fn flatten_block(block: Block, scope: &mut CubeContext) -> Option<ExpandElement> {
    for inner in block.inner {
        flatten_statement(inner, scope);
    }
    block.ret.flatten(scope)
}

fn flatten_tensor_expr(expr: TensorExpression, context: &mut CubeContext) -> Option<ExpandElement> {
    let res = match expr {
        TensorExpression::Stride { tensor, dim } => {
            let tensor = tensor.flatten(context).unwrap().as_variable();
            let dim = dim.flatten(context).unwrap().as_variable();
            let out = context.create_local(Item::new(Elem::UInt));
            context.register(Metadata::Stride {
                dim,
                var: tensor,
                out: out.as_variable(),
            });
            out
        }
        TensorExpression::Shape { tensor, dim } => {
            let tensor = tensor.flatten(context).unwrap().as_variable();
            let dim = dim.flatten(context).unwrap().as_variable();
            let out = context.create_local(Item::new(Elem::UInt));
            context.register(Metadata::Shape {
                dim,
                var: tensor,
                out: out.as_variable(),
            });
            out
        }
        TensorExpression::Length { tensor } => {
            let tensor = tensor.flatten(context).unwrap().as_variable();
            let out = context.create_local(Item::new(Elem::UInt));
            context.register(Metadata::Length {
                var: tensor,
                out: out.clone().into(),
            });
            out
        }
        TensorExpression::Rank { .. } => ExpandElement::Plain(Variable::Rank),
        TensorExpression::Index {
            tensor,
            index,
            vectorization,
        } => {
            let tensor: Variable = tensor.flatten(context).unwrap().into();
            let index: Variable = index.flatten(context).unwrap().into();
            let out = context.create_local(item(tensor.item().elem, vectorization));
            context.register(ir::Operator::Index(BinaryOperator {
                rhs: index,
                lhs: tensor,
                out: out.clone().into(),
            }));
            out
        }
        TensorExpression::Slice { ranges, tensor } => {
            let input = tensor.clone().flatten(context).unwrap().as_variable();
            assert_eq!(ranges.len(), 1, "Multi-slices not currently supported");
            let start = ranges[0]
                .start
                .clone()
                .flatten(context)
                .unwrap()
                .as_variable();
            let end = ranges[0]
                .end
                .clone()
                .and_then(|expr| expr.flatten(context))
                .unwrap_or_else(|| {
                    flatten_tensor_expr(TensorExpression::Length { tensor }, context).unwrap()
                })
                .as_variable();
            let out = context.create_slice(input.item());

            context.register(ir::Operator::Slice(ir::SliceOperator {
                input,
                start,
                end,
                out: out.as_variable(),
            }));

            out
        }
        TensorExpression::__SliceRange(_) => unimplemented!("Slice ranges don't exist at runtime"),
    };
    Some(res)
}

fn flatten_subcube(subcube: SubcubeExpression, context: &mut CubeContext) -> Option<ExpandElement> {
    let res = match subcube {
        SubcubeExpression::Elect => {
            let out = context.create_local(Item::new(subcube.ir_type()));
            context.register(Operation::Subcube(Subcube::Elect(InitOperator {
                out: out.as_variable(),
            })));
            out
        }
        SubcubeExpression::Broadcast {
            left,
            right,
            ty,
            vectorization,
        } => {
            let lhs = left.flatten(context).unwrap().into_variable();
            let rhs = right.flatten(context).unwrap().as_variable();
            let out = context.create_local(item(ty, vectorization));
            context.register(Operation::Subcube(Subcube::Broadcast(BinaryOperator {
                lhs,
                rhs,
                out: out.as_variable(),
            })));
            out
        }
        SubcubeExpression::Unary {
            input,
            operation,
            ty,
        } => {
            let input = input.flatten(context).unwrap().into_variable();
            let out = context.create_local(Item::new(ty));
            let op = map_op(
                operation,
                UnaryOperator {
                    input,
                    out: out.as_variable(),
                },
            );
            context.register(Operation::Subcube(op));
            out
        }
    };
    fn map_op(operation: SubcubeOp, un_op: UnaryOperator) -> Subcube {
        match operation {
            SubcubeOp::All => Subcube::All(un_op),
            SubcubeOp::Any => Subcube::Any(un_op),
            SubcubeOp::Sum => Subcube::Sum(un_op),
            SubcubeOp::Prod => Subcube::Prod(un_op),
            SubcubeOp::Min => Subcube::Min(un_op),
            SubcubeOp::Max => Subcube::Max(un_op),
        }
    }

    Some(res)
}

fn map_bin_op(operator: Operator, bin_op: BinaryOperator) -> ir::Operator {
    match operator {
        Operator::Add => ir::Operator::Add(bin_op),
        Operator::Sub => ir::Operator::Sub(bin_op),
        Operator::Mul => ir::Operator::Mul(bin_op),
        Operator::Div => ir::Operator::Div(bin_op),
        Operator::Rem => ir::Operator::Modulo(bin_op),
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
        Operator::Cos => ir::Operator::Cos(un_op),
        _ => unreachable!("Operator must be unary"),
    }
}

fn split_assign_op(
    left: Expression,
    right: Expression,
    operator: Operator,
    context: &mut CubeContext,
) -> Option<ExpandElement> {
    let new_operator = match operator {
        Operator::AddAssign => Operator::Add,
        Operator::SubAssign => Operator::Sub,
        Operator::MulAssign => Operator::Mul,
        Operator::DivAssign => Operator::Div,
        Operator::RemAssign => Operator::Rem,
        Operator::BitXorAssign => Operator::BitXor,
        Operator::BitAndAssign => Operator::BitAnd,
        Operator::BitOrAssign => Operator::BitOr,
        Operator::ShlAssign => Operator::Shl,
        Operator::ShrAssign => Operator::Shr,
        _ => unreachable!(),
    };
    let (tensor, index) = match left.clone() {
        Expression::Tensor(TensorExpression::Index { tensor, index, .. }) => (tensor, index),
        _ => unreachable!(),
    };
    let binary = {
        let right = right.flatten(context).unwrap().as_variable();
        let left = left.flatten(context).unwrap();
        let operation = map_bin_op(
            new_operator,
            BinaryOperator {
                lhs: left.as_variable(),
                rhs: right,
                out: left.as_variable(),
            },
        );
        context.register(operation);
        left
    };

    let index = index.flatten(context).unwrap().as_variable();
    let tensor = tensor.flatten(context).unwrap().as_variable();
    context.register(ir::Operator::IndexAssign(BinaryOperator {
        lhs: index,
        rhs: binary.into_variable(),
        out: tensor,
    }));
    None
}

pub fn item(ty: Elem, vectorization: Option<NonZero<u8>>) -> Item {
    vectorization
        .map(|vec| Item::vectorized(ty, vec.get()))
        .unwrap_or_else(|| Item::new(ty))
}
