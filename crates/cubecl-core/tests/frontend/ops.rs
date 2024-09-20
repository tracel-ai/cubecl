use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub fn add_op<T: Numeric>(a: T, b: T) -> T {
    a + b
}

#[cube]
pub fn sub_op<T: Numeric>(a: T, b: T) -> T {
    a - b
}

#[cube]
pub fn mul_op<T: Numeric>(a: T, b: T) -> T {
    a * b
}

#[cube]
pub fn div_op<T: Numeric>(a: T, b: T) -> T {
    a / b
}

#[cube]
pub fn abs_op<T: Numeric>(a: T) -> T {
    T::abs(a)
}

#[cube]
pub fn exp_op<F: Float>(a: F) -> F {
    F::exp(a)
}

#[cube]
pub fn log_op<F: Float>(a: F) -> F {
    F::log(a)
}

#[cube]
pub fn log1p_op<F: Float>(a: F) -> F {
    F::log1p(a)
}

#[cube]
pub fn cos_op<F: Float>(a: F) -> F {
    F::cos(a)
}

#[cube]
pub fn sin_op<F: Float>(a: F) -> F {
    F::sin(a)
}

#[cube]
pub fn tanh_op<F: Float>(a: F) -> F {
    F::tanh(a)
}

#[cube]
pub fn powf_op<F: Float>(a: F, b: F) -> F {
    F::powf(a, b)
}

#[cube]
pub fn sqrt_op<F: Float>(a: F) -> F {
    F::sqrt(a)
}

#[cube]
pub fn round_op<F: Float>(a: F) -> F {
    F::round(a)
}

#[cube]
pub fn floor_op<F: Float>(a: F) -> F {
    F::floor(a)
}

#[cube]
pub fn ceil_op<F: Float>(a: F) -> F {
    F::ceil(a)
}

#[cube]
pub fn erf_op<F: Float>(a: F) -> F {
    F::erf(a)
}

#[cube]
pub fn recip_op<F: Float>(a: F) -> F {
    F::recip(a)
}

#[cube]
pub fn equal_op<T: CubePrimitive>(a: T, b: T) -> bool {
    a == b
}

#[cube]
pub fn not_equal_op<T: CubePrimitive>(a: T, b: T) -> bool {
    a != b
}

#[cube]
pub fn lower_op<T: Numeric>(a: T, b: T) -> bool {
    a < b
}

#[cube]
pub fn greater_op<T: Numeric>(a: T, b: T) -> bool {
    a > b
}

#[cube]
pub fn lower_equal_op<T: Numeric>(a: T, b: T) -> bool {
    a <= b
}

#[cube]
pub fn greater_equal_op<T: Numeric>(a: T, b: T) -> bool {
    a >= b
}

#[cube]
pub fn modulo_op(a: u32, b: u32) -> u32 {
    a % b
}

#[cube]
pub fn remainder_op<T: Numeric>(a: T, b: T) -> T {
    T::rem(a, b)
}

#[cube]
pub fn max_op<T: Numeric>(a: T, b: T) -> T {
    T::max(a, b)
}

#[cube]
pub fn min_op<T: Numeric>(a: T, b: T) -> T {
    T::min(a, b)
}

#[cube]
pub fn and_op(a: bool, b: bool) -> bool {
    a && b
}

#[cube]
pub fn or_op(a: bool, b: bool) -> bool {
    a || b
}

#[cube]
pub fn not_op(a: bool) -> bool {
    !a
}

#[cube]
pub fn bitand_op(a: u32, b: u32) -> u32 {
    a & b
}

#[cube]
pub fn bitor_op(a: u32, b: u32) -> u32 {
    a | b
}

#[cube]
pub fn bitxor_op(a: u32, b: u32) -> u32 {
    a ^ b
}

#[cube]
pub fn shl_op(a: u32, b: u32) -> u32 {
    a << b
}

#[cube]
pub fn shr_op(a: u32, b: u32) -> u32 {
    a >> b
}

#[cube]
pub fn add_assign_op<T: Numeric>(mut a: T, b: T) {
    a += b;
}

#[cube]
pub fn sub_assign_op<T: Numeric>(mut a: T, b: T) {
    a -= b;
}

#[cube]
pub fn mul_assign_op<T: Numeric>(mut a: T, b: T) {
    a *= b;
}

#[cube]
pub fn div_assign_op<T: Numeric>(mut a: T, b: T) {
    a /= b;
}

#[cube]
pub fn rem_assign_op<T: Int>(mut a: T, b: T) {
    a %= b;
}

#[cube]
pub fn bitor_assign_op<T: Int>(mut a: T, b: T) {
    a |= b;
}

#[cube]
pub fn bitand_assign_op<T: Int>(mut a: T, b: T) {
    a &= b;
}

#[cube]
pub fn bitxor_assign_op<T: Int>(mut a: T, b: T) {
    a ^= b;
}

#[cube]
pub fn shl_assign_op<T: Int>(mut a: T, b: u32) {
    a <<= b;
}

#[cube]
pub fn shr_assign_op<T: Int>(mut a: T, b: u32) {
    a >>= b;
}

mod tests {
    use super::*;
    use cubecl_core::ir::{Elem, FloatKind, Item};
    use pretty_assertions::assert_eq;

    macro_rules! binary_test {
        ($test_name:ident, $op_expand:expr, $op_name:expr, $func:ident) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::default();
                let x = context.create_local_binding(Item::new(Elem::Float(FloatKind::F32)));
                let y = context.create_local_binding(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand(&mut context, x.into(), y.into());

                assert_eq!(
                    format!("{:?}", context.into_scope().process().operations),
                    $func($op_name)
                );
            }
        };
    }

    macro_rules! unary_test {
        ($test_name:ident, $op_expand:expr, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::default();
                let x = context.create_local_binding(Item::new(Elem::Float(FloatKind::F32)));

                $op_expand(&mut context, x.into());

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_unary($op_name)
                );
            }
        };
    }

    macro_rules! binary_boolean_test {
        ($test_name:ident, $op_expand:expr, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::default();
                let x = context.create_local_binding(Item::new(Elem::Bool));
                let y = context.create_local_binding(Item::new(Elem::Bool));

                $op_expand(&mut context, x.into(), y.into());

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_binary_boolean($op_name)
                );
            }
        };
    }

    macro_rules! binary_u32_test {
        ($test_name:ident, $op_expand:expr, $op_name:expr) => {
            #[test]
            fn $test_name() {
                let mut context = CubeContext::default();
                let x = context.create_local_binding(Item::new(Elem::UInt));
                let y = context.create_local_binding(Item::new(Elem::UInt));

                $op_expand(&mut context, x.into(), y.into());

                assert_eq!(
                    format!("{:?}", context.into_scope().operations),
                    ref_ops_binary_u32($op_name)
                );
            }
        };
    }

    binary_test!(cube_can_add, add_op::expand::<f32>, "Add", ref_ops_binary);
    binary_test!(cube_can_sub, sub_op::expand::<f32>, "Sub", ref_ops_binary);
    binary_test!(cube_can_mul, mul_op::expand::<f32>, "Mul", ref_ops_binary);
    binary_test!(cube_can_div, div_op::expand::<f32>, "Div", ref_ops_binary);
    unary_test!(cube_can_abs, abs_op::expand::<f32>, "Abs");
    unary_test!(cube_can_exp, exp_op::expand::<f32>, "Exp");
    unary_test!(cube_can_log, log_op::expand::<f32>, "Log");
    unary_test!(cube_can_log1p, log1p_op::expand::<f32>, "Log1p");
    unary_test!(cube_can_cos, cos_op::expand::<f32>, "Cos");
    unary_test!(cube_can_sin, sin_op::expand::<f32>, "Sin");
    unary_test!(cube_can_tanh, tanh_op::expand::<f32>, "Tanh");
    binary_test!(
        cube_can_powf,
        powf_op::expand::<f32>,
        "Powf",
        ref_ops_binary
    );
    unary_test!(cube_can_sqrt, sqrt_op::expand::<f32>, "Sqrt");
    unary_test!(cube_can_erf, erf_op::expand::<f32>, "Erf");
    unary_test!(cube_can_recip, recip_op::expand::<f32>, "Recip");
    unary_test!(cube_can_round, round_op::expand::<f32>, "Round");
    unary_test!(cube_can_floor, floor_op::expand::<f32>, "Floor");
    unary_test!(cube_can_ceil, ceil_op::expand::<f32>, "Ceil");
    binary_test!(cube_can_eq, equal_op::expand::<f32>, "Equal", ref_ops_cmp);
    binary_test!(
        cube_can_ne,
        not_equal_op::expand::<f32>,
        "NotEqual",
        ref_ops_cmp
    );
    binary_test!(cube_can_lt, lower_op::expand::<f32>, "Lower", ref_ops_cmp);
    binary_test!(
        cube_can_le,
        lower_equal_op::expand::<f32>,
        "LowerEqual",
        ref_ops_cmp
    );
    binary_test!(
        cube_can_ge,
        greater_equal_op::expand::<f32>,
        "GreaterEqual",
        ref_ops_cmp
    );
    binary_test!(
        cube_can_gt,
        greater_op::expand::<f32>,
        "Greater",
        ref_ops_cmp
    );
    binary_test!(cube_can_max, max_op::expand::<f32>, "Max", ref_ops_binary);
    binary_test!(cube_can_min, min_op::expand::<f32>, "Min", ref_ops_binary);
    binary_test!(
        cube_can_add_assign,
        add_assign_op::expand::<f32>,
        "Add",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_sub_assign,
        sub_assign_op::expand::<f32>,
        "Sub",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_mul_assign,
        mul_assign_op::expand::<f32>,
        "Mul",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_div_assign,
        div_assign_op::expand::<f32>,
        "Div",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_rem_assign,
        rem_assign_op::expand::<i32>,
        "Modulo",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_bitor_assign,
        bitor_assign_op::expand::<i32>,
        "BitwiseOr",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_bitand_assign,
        bitand_assign_op::expand::<i32>,
        "BitwiseAnd",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_bitxor_assign,
        bitxor_assign_op::expand::<i32>,
        "BitwiseXor",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_shl_assign,
        shl_assign_op::expand::<i32>,
        "ShiftLeft",
        ref_ops_binary_assign
    );
    binary_test!(
        cube_can_shr_assign,
        shr_assign_op::expand::<i32>,
        "ShiftRight",
        ref_ops_binary_assign
    );
    binary_boolean_test!(cube_can_and, and_op::expand, "And");
    binary_boolean_test!(cube_can_or, or_op::expand, "Or");
    binary_u32_test!(cube_can_bitand, bitand_op::expand, "BitwiseAnd");
    binary_u32_test!(cube_can_bitor, bitor_op::expand, "BitwiseOr");
    binary_u32_test!(cube_can_bitxor, bitxor_op::expand, "BitwiseXor");
    binary_u32_test!(cube_can_shl, shl_op::expand, "ShiftLeft");
    binary_u32_test!(cube_can_shr, shr_op::expand, "ShiftRight");
    binary_u32_test!(cube_can_mod, modulo_op::expand, "Modulo");
    binary_test!(
        cube_can_rem,
        remainder_op::expand::<f32>,
        "Remainder",
        ref_ops_binary
    );

    #[test]
    fn cube_can_not() {
        let mut context = CubeContext::default();
        let x = context.create_local_binding(Item::new(Elem::Bool));

        not_op::expand(&mut context, x.into());

        assert_eq!(
            format!("{:?}", context.into_scope().operations),
            ref_ops_unary_boolean("Not")
        );
    }

    fn ref_ops_binary_assign(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Float(F32)", "Float(F32)", true, true)
    }

    fn ref_ops_binary(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Float(F32)", "Float(F32)", true, false)
    }

    fn ref_ops_unary(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Float(F32)", "Float(F32)", false, false)
    }

    fn ref_ops_cmp(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Float(F32)", "Bool", true, false)
    }

    fn ref_ops_unary_boolean(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Bool", "Bool", false, false)
    }

    fn ref_ops_binary_boolean(ops_name: &str) -> String {
        ref_ops_template(ops_name, "Bool", "Bool", true, false)
    }

    fn ref_ops_binary_u32(ops_name: &str) -> String {
        ref_ops_template(ops_name, "UInt", "UInt", true, false)
    }

    fn ref_ops_template(
        ops_name: &str,
        in_type: &str,
        out_type: &str,
        binary: bool,
        is_assign: bool,
    ) -> String {
        if binary {
            let out_number = match (in_type == out_type, is_assign) {
                (true, true) => 0,
                (true, false) => binary as i32,
                _ => 2,
            };
            format!(
                "[Operator({ops_name}(BinaryOperator {{ \
                lhs: Local {{ id: 0, item: Item {{ \
                    elem: {in_type}, \
                    vectorization: None \
                }}, depth: 0 }}, \
                rhs: Local {{ id: 1, item: Item {{ \
                    elem: {in_type}, \
                    vectorization: None \
                }}, depth: 0 }}, \
                out: Local {{ id: {out_number}, item: Item {{ \
                    elem: {out_type}, \
                    vectorization: None \
                }}, depth: 0 }} \
            }}))]"
            )
        } else {
            format!(
                "[Operator({ops_name}(UnaryOperator {{ \
                input: Local {{ id: 0, item: Item {{ \
                    elem: {in_type}, \
                    vectorization: None \
                }}, depth: 0 }}, \
                out: Local {{ id: 0, item: Item {{ \
                    elem: {out_type}, \
                    vectorization: None \
                }}, depth: 0 }} \
            }}))]"
            )
        }
    }
}
