use std::num::NonZero;

use cubecl_core::{
    ir::{Elem, Item},
    new_ir::{Backend, CubeType, NewExpr, Operator, Vectorization},
    prelude::{CubeContext, ExpandElement},
};

use crate::compiler::wgsl::{Instruction, WgslCompiler};

macro_rules! e {
    ($ty:path) => {
        impl NewExpr<Self, Output = $ty>
    };
}

pub struct WgpuBackend {
    context: CubeContext,
    compiler: WgslCompiler,
    instructions: Vec<Instruction>,
}

impl Backend for WgpuBackend {
    fn expand_binop<Left: CubeType, Right: CubeType>(
        &mut self,
        left: &e!(Left),
        right: &e!(Right),
        op: Operator,
        ty: Elem,
        vectorization: Vectorization,
    ) -> ExpandElement {
        let left = left.expand(self);
        let right = right.expand(self);
        let right = right.into_variable();

        let (left, out) = if op.is_assign() {
            (left.as_variable(), left)
        } else {
            (
                left.into_variable(),
                self.context.create_local(item(ty, vectorization)),
            )
        };

        self.instructions.push(Instruction::Add {
            lhs: self.compiler.compile_variable(left),
            rhs: self.compiler.compile_variable(right),
            out: self.compiler.compile_variable(out.as_variable()),
        });

        out
    }
}

pub fn item(ty: Elem, vectorization: Option<NonZero<u8>>) -> Item {
    vectorization
        .map(|vec| Item::vectorized(ty, vec.get()))
        .unwrap_or_else(|| Item::new(ty))
}
