use cubecl_ir::{
    Arithmetic, BinaryOperands, Comparison, ElemType, IndexOperands, Instruction, Memory,
    Operation, Scope, Type, UnaryOperands, Value, VectorSize,
};
use cubecl_macros::cube;

use crate::{
    self as cubecl,
    frontend::{validate_complex_assign_operation, validate_complex_operation},
    prelude::*,
};

pub(crate) fn read_value(scope: &Scope, val: Value) -> Value {
    if let Type::Pointer(inner, _) = val.ty {
        let out = scope.create_value(*inner);
        scope.register(Instruction::new(Memory::Load(val), out));
        out
    } else {
        val
    }
}

pub(crate) fn binary_expand<F, Op>(scope: &Scope, lhs: Value, rhs: Value, func: F) -> Value
where
    F: Fn(BinaryOperands) -> Op,
    Op: Into<Operation>,
{
    let item_lhs = lhs.value_type();
    let item_rhs = rhs.value_type();

    let vector_size = find_vectorization(item_lhs, item_rhs);

    let item = item_lhs.with_vector_size(vector_size);

    let output = scope.create_value(item);

    let op = func(BinaryOperands { lhs, rhs }).into();
    validate_complex_operation(scope, &op);

    scope.register(Instruction::new(op, output));

    output
}

pub(crate) fn index_expand(scope: &Scope, list: Value, index: Value, checked: bool) -> Value {
    let ty = list.value_type();

    let class = list.address_space();
    let output = scope.create_value(Type::pointer(ty, class));

    let op = Memory::Index(IndexOperands {
        list,
        index,
        unroll_factor: 1,
        checked,
    });

    scope.register(Instruction::new(op, output));

    output
}

pub(crate) fn binary_expand_fixed_output<F>(
    scope: &Scope,
    lhs: Value,
    rhs: Value,
    out_item: Type,
    func: F,
) -> Value
where
    F: Fn(BinaryOperands) -> Arithmetic,
{
    let out = scope.create_value(out_item);
    let op = func(BinaryOperands { lhs, rhs }).into();
    validate_complex_operation(scope, &op);

    scope.register(Instruction::new(op, out));

    out
}

pub(crate) fn cmp_expand<F>(scope: &Scope, lhs: Value, rhs: Value, func: F) -> Value
where
    F: Fn(BinaryOperands) -> Comparison,
{
    let item_lhs = lhs.value_type();
    let item_rhs = rhs.value_type();

    let vector_size = find_vectorization(item_lhs, item_rhs);

    let out_item = Type::scalar(ElemType::Bool).with_vector_size(vector_size);

    let out = scope.create_value(out_item);

    let op = func(BinaryOperands { lhs, rhs }).into();
    validate_complex_operation(scope, &op);

    scope.register(Instruction::new(op, out));

    out
}

pub(crate) fn assign_op_expand<T: CubeType, Op>(
    scope: &Scope,
    lhs: &mut NativeExpand<T>,
    rhs: NativeExpand<T>,
    func: impl Fn(BinaryOperands) -> Op,
) where
    Op: Into<Operation>,
    NativeExpand<T>: DerefExpand<Target = NativeExpand<T>>,
{
    let lhs_value = lhs.__expand_deref_method(scope).expand;
    let lhs = lhs.expand;
    let rhs = rhs.expand;

    if lhs.is_immutable() {
        panic!("Can't have a mutable operation on a const variable. Try to use `RuntimeCell`.");
    }

    let tmp = scope.create_value(lhs.value_type());
    let op = func(BinaryOperands {
        lhs: lhs_value,
        rhs,
    })
    .into();
    validate_complex_assign_operation(scope, &op);

    scope.register(Instruction::new(op, tmp));
    assign::expand_element(scope, tmp, lhs);
}

pub fn unary_expand<F, Op>(scope: &Scope, input: Value, func: F) -> Value
where
    F: Fn(UnaryOperands) -> Op,
    Op: Into<Operation>,
{
    let item = input.value_type();

    let out = scope.create_value(item);

    let op = func(UnaryOperands { input }).into();
    validate_complex_operation(scope, &op);

    scope.register(Instruction::new(op, out));

    out
}

pub fn unary_expand_fixed_output<F, Op>(
    scope: &Scope,
    input: Value,
    out_item: Type,
    func: F,
) -> Value
where
    F: Fn(UnaryOperands) -> Op,
    Op: Into<Operation>,
{
    let output = scope.create_value(out_item);

    let op = func(UnaryOperands { input }).into();
    validate_complex_operation(scope, &op);

    scope.register(Instruction::new(op, output));

    output
}

pub fn init_expand(scope: &Scope, input: Value, mutable: bool) -> Value {
    let input = read_value(scope, input);
    let ty = input.ty;

    let out = if mutable {
        scope.create_local_mut(ty)
    } else {
        scope.create_value(ty)
    };

    assign::expand_element(scope, input, out);

    out
}

pub(crate) fn find_vectorization(lhs: Type, rhs: Type) -> VectorSize {
    if matches!(lhs, Type::Scalar(_)) && matches!(rhs, Type::Scalar(_)) {
        0
    } else {
        lhs.vector_size().max(rhs.vector_size())
    }
}

pub fn assign_binary_op_expand<
    A: CubeType,
    V: CubeType,
    F: Fn(BinaryOperands) -> Op,
    Op: Into<Operation>,
>(
    scope: &Scope,
    lhs: &mut NativeExpand<A>,
    rhs: NativeExpand<V>,
    func: F,
) where
    NativeExpand<A>: DerefExpand<Target = NativeExpand<A>> + Assign,
{
    let lhs_value = lhs.__expand_deref_method(scope).expand;
    let rhs: Value = rhs.into();
    let out = scope.create_value(lhs.expand.ty);

    let op = func(BinaryOperands {
        lhs: lhs_value,
        rhs,
    })
    .into();
    validate_complex_assign_operation(scope, &op);
    scope.register(Instruction::new(op, out));
    lhs.__expand_assign_method(scope, out.into());
}

pub trait DivCeil: Int + CubeType<ExpandType: DivCeilExpand<Self>> {
    fn div_ceil(self, divisor: Self) -> Self;

    fn __expand_div_ceil(
        scope: &Scope,
        a: NativeExpand<Self>,
        b: NativeExpand<Self>,
    ) -> NativeExpand<Self> {
        a.__expand_div_ceil_method(scope, b)
    }
}

pub trait DivCeilExpand<E: Int> {
    fn __expand_div_ceil_method(self, scope: &Scope, divisor: Self) -> Self;
}

impl<E: DivCeil> DivCeilExpand<E> for NativeExpand<E> {
    fn __expand_div_ceil_method(self, scope: &Scope, divisor: NativeExpand<E>) -> NativeExpand<E> {
        div_ceil::expand::<E>(scope, self, divisor)
    }
}

macro_rules! impl_div_ceil {
    ($($ty:ty),*) => {
        $(
            impl DivCeil for $ty {
                #[allow(clippy::manual_div_ceil)] // Need to define div_ceil to use div_ceil!
                fn div_ceil(self, divisor: Self) -> Self {
                    (self + divisor - 1) / divisor
                }
            }
        )*
    };
}

impl_div_ceil!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

impl<E: Int> NativeExpand<E> {
    pub fn __expand_is_multiple_of_method(
        self,
        scope: &Scope,
        factor: NativeExpand<E>,
    ) -> NativeExpand<bool> {
        let modulo = self.__expand_rem_method(scope, factor);
        modulo.__expand_eq_method(scope, &E::from_int(0).into())
    }
}

#[cube]
pub fn div_ceil<E: Int>(a: E, b: E) -> E {
    (a + b - E::new(1)) / b
}
