use crate::ir::{BinaryOperator, Elem, Item, Operator, UnaryOperator, Variable, Vectorization};
use crate::prelude::{CubeType, ExpandElementTyped};
use crate::{
    frontend::{CubeContext, ExpandElement},
    prelude::CubeIndex,
};

pub(crate) fn binary_expand<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs = lhs.consume();
    let rhs = rhs.consume();

    let item_lhs = lhs.item();
    let item_rhs = rhs.item();

    let vectorization = find_vectorization(item_lhs.vectorization, item_rhs.vectorization);

    let item = Item::vectorized(item_lhs.elem, vectorization);

    let output = context.create_local_binding(item);
    let out = *output;

    let op = func(BinaryOperator { lhs, rhs, out });

    context.register(op);

    output
}

pub(crate) fn binary_expand_fixed_output<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    out_item: Item,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs_var = lhs.consume();
    let rhs_var = rhs.consume();

    let out = context.create_local_binding(out_item);

    let out_var = *out;

    let op = func(BinaryOperator {
        lhs: lhs_var,
        rhs: rhs_var,
        out: out_var,
    });

    context.register(op);

    out
}

pub(crate) fn binary_expand_no_vec<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs = lhs.consume();
    let rhs = rhs.consume();

    let item_lhs = lhs.item();

    let item = Item::new(item_lhs.elem);

    let output = context.create_local_binding(item);
    let out = *output;

    let op = func(BinaryOperator { lhs, rhs, out });

    context.register(op);

    output
}

pub(crate) fn cmp_expand<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs: Variable = *lhs;
    let rhs: Variable = *rhs;
    let item = lhs.item();

    find_vectorization(item.vectorization, rhs.item().vectorization);

    let out_item = Item {
        elem: Elem::Bool,
        vectorization: item.vectorization,
    };

    let out = context.create_local_binding(out_item);
    let out_var = *out;

    let op = func(BinaryOperator {
        lhs,
        rhs,
        out: out_var,
    });

    context.register(op);

    out
}

pub(crate) fn assign_op_expand<F>(
    context: &mut CubeContext,
    lhs: ExpandElement,
    rhs: ExpandElement,
    func: F,
) -> ExpandElement
where
    F: Fn(BinaryOperator) -> Operator,
{
    let lhs_var: Variable = *lhs;
    let rhs: Variable = *rhs;

    find_vectorization(lhs_var.item().vectorization, rhs.item().vectorization);

    let op = func(BinaryOperator {
        lhs: lhs_var,
        rhs,
        out: lhs_var,
    });

    context.register(op);

    lhs
}

pub fn unary_expand<F>(context: &mut CubeContext, input: ExpandElement, func: F) -> ExpandElement
where
    F: Fn(UnaryOperator) -> Operator,
{
    let input = input.consume();
    let item = input.item();

    let out = context.create_local_binding(item);
    let out_var = *out;

    let op = func(UnaryOperator {
        input,
        out: out_var,
    });

    context.register(op);

    out
}

pub fn unary_expand_fixed_output<F>(
    context: &mut CubeContext,
    input: ExpandElement,
    out_item: Item,
    func: F,
) -> ExpandElement
where
    F: Fn(UnaryOperator) -> Operator,
{
    let input = input.consume();
    let output = context.create_local_binding(out_item);
    let out = *output;

    let op = func(UnaryOperator { input, out });

    context.register(op);

    output
}

pub fn init_expand<F>(context: &mut CubeContext, input: ExpandElement, func: F) -> ExpandElement
where
    F: Fn(UnaryOperator) -> Operator,
{
    if input.can_mut() {
        return input;
    }

    let input_var: Variable = *input;
    let item = input.item();

    let out = context.create_local_variable(item);
    let out_var = *out;

    let op = func(UnaryOperator {
        input: input_var,
        out: out_var,
    });

    context.register(op);

    out
}

fn find_vectorization(lhs: Vectorization, rhs: Vectorization) -> Vectorization {
    match (lhs, rhs) {
        (None, None) => None,
        (None, Some(rhs)) => Some(rhs),
        (Some(lhs), None) => Some(lhs),
        (Some(lhs), Some(rhs)) if lhs == rhs => Some(lhs),
        (Some(lhs), Some(rhs)) => {
            panic!(
                "Left and right have different vectorizations.
                Left: {lhs}, right: {rhs}.
                Auto-matching fixed vectorization currently unsupported."
            );
        }
    }
}

pub fn array_assign_binary_op_expand<
    A: CubeType + CubeIndex<u32>,
    V: CubeType,
    F: Fn(BinaryOperator) -> Operator,
>(
    context: &mut CubeContext,
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

    let array_value = context.create_local_binding(array.item());

    let read = Operator::Index(BinaryOperator {
        lhs: *array,
        rhs: *index,
        out: *array_value,
    });
    let array_value = array_value.consume();
    let op_out = context.create_local_binding(array.item());
    let calculate = func(BinaryOperator {
        lhs: array_value,
        rhs: *value,
        out: *op_out,
    });

    let write = Operator::IndexAssign(BinaryOperator {
        lhs: *index,
        rhs: op_out.consume(),
        out: *array,
    });

    context.register(read);
    context.register(calculate);
    context.register(write);
}
