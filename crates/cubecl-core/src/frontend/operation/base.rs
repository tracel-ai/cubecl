use std::num::NonZeroU8;

use cubecl_ir::{
    Arithmetic, BinaryOperator, Comparison, Elem, ExpandElement, Instruction, Item, Operation,
    Operator, Scope, UnaryOperator, Variable, VariableKind, Vectorization,
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

    let item_lhs = lhs.item;
    let item_rhs = rhs.item;

    let vectorization = find_vectorization(item_lhs.vectorization, item_rhs.vectorization);

    let item = Item::vectorized(item_lhs.elem, vectorization);

    let output = scope.create_local(item);
    let out = *output;

    let op = func(BinaryOperator { lhs, rhs });

    scope.register(Instruction::new(op, out));

    output
}

pub(crate) fn binary_expand_fixed_output<F>(
    scope: &mut Scope,
    lhs: ExpandElement,
    rhs: ExpandElement,
    out_item: Item,
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

pub(crate) fn binary_expand_no_vec<F>(
    scope: &mut Scope,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs = lhs.consume();
    let rhs = rhs.consume();

    let item_lhs = lhs.item;

    let item = Item::new(item_lhs.elem);

    let output = scope.create_local(item);
    let out = *output;

    let op = func(BinaryOperator { lhs, rhs });

    scope.register(Instruction::new(op, out));

    output
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
    let item = lhs.item;

    find_vectorization(item.vectorization, rhs.item.vectorization);

    let out_item = Item {
        elem: Elem::Bool,
        vectorization: item.vectorization,
    };

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
    let lhs_var: Variable = *lhs;
    let rhs: Variable = *rhs;

    find_vectorization(lhs_var.item.vectorization, rhs.item.vectorization);

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
    let item = input.item;

    let out = scope.create_local(item);
    let out_var = *out;

    let op = func(UnaryOperator { input });

    scope.register(Instruction::new(op, out_var));

    out
}

pub fn unary_expand_fixed_output<F, Op>(
    scope: &mut Scope,
    input: ExpandElement,
    out_item: Item,
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

pub fn init_expand<F>(scope: &mut Scope, input: ExpandElement, func: F) -> ExpandElement
where
    F: Fn(Variable) -> Operation,
{
    if input.can_mut() {
        return input;
    }
    let input_var: Variable = *input;
    let item = input.item;

    let out = scope.create_local_mut(item); // TODO: The mut is safe, but unnecessary if the variable is immutable.
    let out_var = *out;

    let op = func(input_var);
    scope.register(Instruction::new(op, out_var));

    out
}

fn find_vectorization(lhs: Vectorization, rhs: Vectorization) -> Vectorization {
    match (lhs, rhs) {
        (None, None) => None,
        (None, Some(rhs)) => Some(rhs),
        (Some(lhs), None) => Some(lhs),
        (Some(lhs), Some(rhs)) => {
            if lhs == rhs {
                Some(lhs)
            } else if lhs == NonZeroU8::new(1).unwrap() || rhs == NonZeroU8::new(1).unwrap() {
                Some(core::cmp::max(lhs, rhs))
            } else {
                panic!(
                    "Left and right have different vectorizations.
                    Left: {lhs}, right: {rhs}.
                    Auto-matching fixed vectorization currently unsupported."
                );
            }
        }
    }
}

pub fn array_assign_binary_op_expand<
    A: CubeType + CubeIndex<u32>,
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
        VariableKind::LocalMut { .. } => array.item.vectorize(None),
        _ => array.item,
    };
    let array_value = scope.create_local(array_item);

    let read = Instruction::new(
        Operator::Index(BinaryOperator {
            lhs: *array,
            rhs: *index,
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

    let write = Operator::IndexAssign(BinaryOperator {
        lhs: *index,
        rhs: op_out.consume(),
    });
    scope.register(read);
    scope.register(calculate);
    scope.register(Instruction::new(write, *array));
}
