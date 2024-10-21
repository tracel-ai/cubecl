use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
struct SimpleType {
    a: u32,
}

#[cube]
impl SimpleType {
    #[allow(dead_code)]
    fn simple_method(&self, lhs: u32) -> u32 {
        self.a * lhs
    }

    #[allow(dead_code)]
    pub fn call_method_inner(&self) -> u32 {
        self.simple_method(5u32)
    }

    #[allow(dead_code)]
    pub fn call_method_as_function_inner(&self) -> u32 {
        Self::simple_method(self, 5u32)
    }

    #[allow(dead_code)]
    pub fn return_self(self) -> Self {
        self
    }

    #[allow(dead_code)]
    pub fn with_other(self, other: Self) -> u32 {
        self.call_method_inner() + other.call_method_inner()
    }

    #[allow(dead_code)]
    pub fn with_generic<E: Float>(self, rhs: E) -> u32 {
        self.simple_method(u32::cast_from(rhs))
    }
}

#[derive(CubeType)]
struct TypeGeneric<C: CubePrimitive> {
    a: C,
}

#[cube]
impl<C: Numeric> TypeGeneric<C> {
    #[allow(dead_code)]
    fn value(&self, lhs: u32) -> C {
        self.a * C::cast_from(lhs)
    }

    #[allow(dead_code)]
    pub fn call_inner(&self) -> C {
        let val1 = self.value(5u32);
        let val2 = Self::value(self, 2u32);
        val1 + val2
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
    pub fn complex_method(&mut self, lhs: f32, rhs: C) -> f32 {
        let tmp = self.a + (C::cast_from(lhs) / rhs);

        Self::simple_function(lhs, tmp)
    }

    fn simple_function(lhs: f32, rhs: C) -> f32 {
        lhs * f32::cast_from(rhs)
    }
}
