use half::{bf16, f16};

use crate::{
    flex32,
    ir::{Arithmetic, ClampOperator, ExpandElement},
    prelude::{CubeContext, CubePrimitive},
    tf32, unexpanded,
};

use super::unary_expand;

pub trait Clamp: CubePrimitive + Sized {
    /// Clamp the input value between the max and min values provided.
    #[allow(unused_variables)]
    fn clamp(input: Self, min_value: Self, max_value: Self) -> Self {
        unexpanded!()
    }
    fn __expand_clamp(
        context: &mut CubeContext,
        input: Self::ExpandType,
        min_value: Self::ExpandType,
        max_value: Self::ExpandType,
    ) -> Self::ExpandType {
        let input: ExpandElement = input.into();
        let min_value: ExpandElement = min_value.into();
        let max_value: ExpandElement = max_value.into();

        unary_expand(context, input, |op| {
            Arithmetic::Clamp(ClampOperator {
                input: op.input,
                min_value: *min_value,
                max_value: *max_value,
            })
        })
        .into()
    }
}

impl Clamp for f16 {}
impl Clamp for bf16 {}
impl Clamp for flex32 {}
impl Clamp for tf32 {}
impl Clamp for f32 {}
impl Clamp for f64 {}
impl Clamp for i8 {}
impl Clamp for i16 {}
impl Clamp for i32 {}
impl Clamp for i64 {}
impl Clamp for u8 {}
impl Clamp for u16 {}
impl Clamp for u32 {}
impl Clamp for u64 {}
