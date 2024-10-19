use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
struct SimpleType {
    a: u32,
}

#[cube]
#[allow(dead_code)]
impl SimpleType {
    #[allow(dead_code)]
    fn value(self, lhs: u32) -> u32 {
        self.a * lhs
    }

    #[allow(dead_code)]
    pub fn with_five(self) -> u32 {
        let val = self.value(5u32);
        val
    }
}

#[derive(CubeType)]
struct TypeGeneric<C: CubePrimitive> {
    a: C,
}

#[cube]
impl<C: Numeric> TypeGeneric<C> {
    /// My docs.
    #[allow(dead_code)]
    fn value(&self, lhs: u32) -> C {
        self.a * C::cast_from(lhs)
    }

    #[allow(dead_code)]
    pub fn with_five(&self) -> C {
        let val = self.value(5u32);
        val
    }
}

#[derive(CubeType)]
struct ComplexType<C: Numeric, T: Numeric> {
    a: C,
    t: T,
}

#[cube]
impl<C: Numeric> ComplexType<C, f32> {
    #[allow(dead_code)]
    pub fn execute_suff(&mut self, lhs: f32, rhs: C) -> f32 {
        let tmp = self.a + (C::cast_from(lhs) / rhs);

        ComplexType::<C, f32>::functional(lhs, tmp)
    }

    fn functional(lhs: f32, rhs: C) -> f32 {
        lhs * f32::cast_from(rhs)
    }
}
