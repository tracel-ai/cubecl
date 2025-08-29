use cubecl_ir::{
    Arithmetic, BinaryOperator, Comparison, ElemType, ExpandElement, IndexAssignOperator,
    IndexOperator, Instruction, LineSize, Operation, Operator, Scope, Type, UnaryOperator,
    Variable, VariableKind,
};

use crate::prelude::{CubeIndex, CubeType, ExpandElementTyped};

pub(crate) fn binary_expand<F, Op>(
    scope: &mut Scope,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Op,
    Op: Into<Operation>,
{
    let lhs = lhs.consume();
    let rhs = rhs.consume();

    let item_lhs = lhs.ty;
    let item_rhs = rhs.ty;

    let line_size = find_vectorization(item_lhs, item_rhs);

    let item = item_lhs.line(line_size);

    let output = scope.create_local(item);
    let out = *output;

    let op = func(BinaryOperator { lhs, rhs });

    scope.register(Instruction::new(op, out));

    output
}

pub(crate) fn index_expand_no_vec<F>(
    scope: &mut Scope,
    list: ExpandElement,
    index: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(IndexOperator) -> Operator,
{
    let list = list.consume();
    let index = index.consume();

    let item_lhs = list.ty;

    let item = item_lhs.line(0);

    let output = scope.create_local(item);
    let out = *output;

    let op = func(IndexOperator {
        list,
        index,
        line_size: 0u32,
        unroll_factor: 1,
    });

    scope.register(Instruction::new(op, out));

    output
}
pub(crate) fn index_expand<F, Op>(
    scope: &mut Scope,
    list: ExpandElement,
    index: ExpandElement,
    line_size: Option<u32>,
    func: F,
) -> ExpandElement
where
    F: Fn(IndexOperator) -> Op,
    Op: Into<Operation>,
{
    let list = list.consume();
    let index = index.consume();

    let item_lhs = list.ty;
    let item_rhs = index.ty;

    let vec = if let Some(line_size) = line_size {
        line_size
    } else {
        find_vectorization(item_lhs, item_rhs)
    };

    let item = item_lhs.line(vec);

    let output = scope.create_local(item);
    let out = *output;

    let op = func(IndexOperator {
        list,
        index,
        line_size: line_size.unwrap_or(0),
        unroll_factor: 1,
    });

    scope.register(Instruction::new(op, out));

    output
}

pub(crate) fn binary_expand_fixed_output<F>(
    scope: &mut Scope,
    lhs: ExpandElement,
    rhs: ExpandElement,
    out_item: Type,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Arithmetic,
{
    let lhs_var = lhs.consume();
    let rhs_var = rhs.consume();

    let out = scope.create_local(out_item);

    let out_var = *out;

    let op = func(BinaryOperator {
        lhs: lhs_var,
        rhs: rhs_var,
    });

    scope.register(Instruction::new(op, out_var));

    out
}

pub(crate) fn cmp_expand<F>(
    scope: &mut Scope,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Comparison,
{
    let lhs: Variable = *lhs;
    let rhs: Variable = *rhs;
    let item = lhs.ty;

    let out_item = Type::scalar(ElemType::Bool).line(item.line_size());

    let out = scope.create_local(out_item);
    let out_var = *out;

    let op = func(BinaryOperator { lhs, rhs });

    scope.register(Instruction::new(op, out_var));

    out
}

pub(crate) fn assign_op_expand<F, Op>(
    scope: &mut Scope,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Op,
    Op: Into<Operation>,
{
    if lhs.is_immutable() {
        panic!("Can't have a mutable operation on a const variable. Try to use `RuntimeCell`.");
    }
    let lhs_var: Variable = *lhs;
    let rhs: Variable = *rhs;

    let op = func(BinaryOperator { lhs: lhs_var, rhs });

    scope.register(Instruction::new(op, lhs_var));

    lhs
}

pub fn unary_expand<F, Op>(scope: &mut Scope, input: ExpandElement, func: F) -> ExpandElement
where
    F: Fn(UnaryOperator) -> Op,
    Op: Into<Operation>,
{
    let input = input.consume();
    let item = input.ty;

    let out = scope.create_local(item);
    let out_var = *out;

    let op = func(UnaryOperator { input });

    scope.register(Instruction::new(op, out_var));

    out
}

pub fn unary_expand_fixed_output<F, Op>(
    scope: &mut Scope,
    input: ExpandElement,
    out_item: Type,
    func: F,
) -> ExpandElement
where
    F: Fn(UnaryOperator) -> Op,
    Op: Into<Operation>,
{
    let input = input.consume();
    let output = scope.create_local(out_item);
    let out = *output;

    let op = func(UnaryOperator { input });

    scope.register(Instruction::new(op, out));

    output
}

pub fn init_expand<F>(
    scope: &mut Scope,
    input: ExpandElement,
    mutable: bool,
    func: F,
) -> ExpandElement
where
    F: Fn(Variable) -> Operation,
{
    let input_var: Variable = *input;
    let item = input.ty;

    let out = if mutable {
        scope.create_local_mut(item)
    } else {
        scope.create_local(item)
    };

    let out_var = *out;

    let op = func(input_var);
    scope.register(Instruction::new(op, out_var));

    out
}

fn find_vectorization(lhs: Type, rhs: Type) -> LineSize {
    if matches!(lhs, Type::Scalar(_)) && matches!(rhs, Type::Scalar(_)) {
        0
    } else {
        lhs.line_size().max(rhs.line_size())
    }
}

pub fn array_assign_binary_op_expand<
    A: CubeType + CubeIndex,
    V: CubeType,
    F: Fn(BinaryOperator) -> Op,
    Op: Into<Operation>,
>(
    scope: &mut Scope,
    array: ExpandElementTyped<A>,
    index: ExpandElementTyped<u32>,
    value: ExpandElementTyped<V>,
    func: F,
) where
    A::Output: CubeType + Sized,
{
    let array: ExpandElement = array.into();
    let index: ExpandElement = index.into();
    let value: ExpandElement = value.into();

    let array_item = match array.kind {
        // In that case, the array is a line.
        VariableKind::LocalMut { .. } => array.ty.line(0),
        _ => array.ty,
    };
    let array_value = scope.create_local(array_item);

    let read = Instruction::new(
        Operator::Index(IndexOperator {
            list: *array,
            index: *index,
            line_size: 0,
            unroll_factor: 1,
        }),
        *array_value,
    );
    let array_value = array_value.consume();
    let op_out = scope.create_local(array_item);
    let calculate = Instruction::new(
        func(BinaryOperator {
            lhs: array_value,
            rhs: *value,
        }),
        *op_out,
    );

    let write = Operator::IndexAssign(IndexAssignOperator {
        index: *index,
        value: op_out.consume(),
        line_size: 0,
        unroll_factor: 1,
    });
    scope.register(read);
    scope.register(calculate);
    scope.register(Instruction::new(write, *array));
}
