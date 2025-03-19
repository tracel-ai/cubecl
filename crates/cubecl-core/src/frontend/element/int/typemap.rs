use bytemuck::{Pod, Zeroable};
use core::ops::*;
use cubecl_ir::{Elem, ExpandElement, IntKind, Scope, Variable};
use derive_more::derive::*;
use num_traits::{NumCast, ToPrimitive};
use serde::Serialize;

use crate::{
    Runtime,
    compute::{KernelBuilder, KernelLauncher},
    prelude::Index,
    prelude::*,
};

use super::{Int, init_expand_element};

#[repr(transparent)]
#[derive(
    Clone,
    Copy,
    Default,
    Serialize,
    Zeroable,
    Pod,
    PartialEq,
    PartialOrd,
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    AddAssign,
    SubAssign,
    MulAssign,
    DivAssign,
    RemAssign,
    Debug,
    Display,
    Shl,
    ShlAssign,
    Shr,
    ShrAssign,
    BitXor,
    BitXorAssign,
    BitAnd,
    BitAndAssign,
    BitOr,
    BitOrAssign,
    Not,
)]
pub struct IntExpand<const POS: u8>(i64);

impl<const POS: u8> Mul for IntExpand<POS> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 * rhs.0)
    }
}

impl<const POS: u8> Div for IntExpand<POS> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 / rhs.0)
    }
}

impl<const POS: u8> Rem for IntExpand<POS> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 % rhs.0)
    }
}

impl<const POS: u8> MulAssign for IntExpand<POS> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl<const POS: u8> DivAssign for IntExpand<POS> {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl<const POS: u8> RemAssign for IntExpand<POS> {
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl<const POS: u8> Shr for IntExpand<POS> {
    type Output = Self;

    fn shr(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 >> rhs.0)
    }
}

impl<const POS: u8> Shl for IntExpand<POS> {
    type Output = Self;

    fn shl(self, rhs: Self) -> Self::Output {
        IntExpand(self.0 << rhs.0)
    }
}

impl<const POS: u8> ToPrimitive for IntExpand<POS> {
    fn to_i64(&self) -> Option<i64> {
        Some(self.0)
    }

    fn to_u64(&self) -> Option<u64> {
        Some(self.0 as u64)
    }

    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }

    fn to_f64(&self) -> Option<f64> {
        Some(self.0 as f64)
    }
}

impl<const POS: u8> NumCast for IntExpand<POS> {
    fn from<T: num_traits::ToPrimitive>(n: T) -> Option<Self> {
        Some(IntExpand(n.to_i64()?))
    }
}

impl<const POS: u8> CubeType for IntExpand<POS> {
    type ExpandType = ExpandElementTyped<IntExpand<POS>>;
}

impl<const POS: u8> CubePrimitive for IntExpand<POS> {
    /// Return the element type to use on GPU
    fn as_elem(scope: &Scope) -> Elem {
        scope.resolve_elem::<Self>().expect("Type to be registered")
    }
}

impl<const POS: u8> From<IntExpand<POS>> for Variable {
    fn from(val: IntExpand<POS>) -> Self {
        // TODO: Fix how we create literal.
        Variable::new(
            crate::ir::VariableKind::ConstantScalar(crate::ir::ConstantScalarValue::Int(
                val.0,
                cubecl_ir::IntKind::I32,
            )),
            crate::ir::Item::new(Elem::Int(IntKind::I64)),
        )
    }
}

impl<const POS: u8> From<IntExpand<POS>> for ExpandElementTyped<IntExpand<POS>> {
    fn from(value: IntExpand<POS>) -> Self {
        let var: Variable = value.into();
        ExpandElementTyped::new(ExpandElement::Plain(var))
    }
}

impl<const POS: u8> IntoRuntime for IntExpand<POS> {
    fn __expand_runtime_method(self, scope: &mut Scope) -> ExpandElementTyped<Self> {
        let expand: ExpandElementTyped<Self> = ExpandElementTyped::from_lit(scope, self.0);
        Init::init(expand, scope)
    }
}

impl<const POS: u8> Numeric for IntExpand<POS> {
    fn min_value() -> Self {
        panic!("Can't use min value in comptime with dynamic element type");
    }
    fn max_value() -> Self {
        panic!("Can't use max value in comptime with dynamic element type");
    }
}

impl<const POS: u8> ExpandElementBaseInit for IntExpand<POS> {
    fn init_elem(scope: &mut Scope, elem: ExpandElement) -> ExpandElement {
        init_expand_element(scope, elem)
    }
}

impl<const POS: u8> Remainder for IntExpand<POS> {}
impl<const POS: u8> Abs for IntExpand<POS> {}
impl<const POS: u8> Max for IntExpand<POS> {}
impl<const POS: u8> Min for IntExpand<POS> {}
impl<const POS: u8> Clamp for IntExpand<POS> {}

impl<const POS: u8> BitwiseNot for IntExpand<POS> {}
impl<const POS: u8> ReverseBits for IntExpand<POS> {}
impl<const POS: u8> CountOnes for IntExpand<POS> {}
impl<const POS: u8> FindFirstSet for IntExpand<POS> {}
impl<const POS: u8> LeadingZeros for IntExpand<POS> {}

impl<T: Index, const POS: u8> CubeIndex<T> for IntExpand<POS> {
    type Output = Self;
}
impl<T: Index, const POS: u8> CubeIndexMut<T> for IntExpand<POS> {}

impl<const POS: u8> Int for IntExpand<POS> {
    const BITS: u32 = 32;

    fn new(val: i64) -> Self {
        IntExpand(val)
    }
}

impl<const POS: u8> LaunchArgExpand for IntExpand<POS> {
    type CompilationArg = ();

    fn expand(_: &Self::CompilationArg, builder: &mut KernelBuilder) -> ExpandElementTyped<Self> {
        builder
            .scalar(IntExpand::<POS>::as_elem(&builder.context))
            .into()
    }
}

impl<const POS: u8> ScalarArgSettings for IntExpand<POS> {
    fn register<R: Runtime>(&self, settings: &mut KernelLauncher<R>) {
        settings.register_i32(self.0 as i32);
    }
}
