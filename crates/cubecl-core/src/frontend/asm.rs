use alloc::{string::String, vec::Vec};

use cubecl_ir::{
    Scope,
    dialect::{InlineAsmOp, OperationPtrExt},
    interfaces::TypeExt,
};
use pliron::{op::Op, r#type::Typed, value::Value};

use crate::frontend::{HasValue, assign};

#[derive(Default)]
pub struct BuildAsmExpand {
    asm: String,
    out_values: Vec<Value>,
    in_values: Vec<Value>,
    pure: bool,
    nomem: bool,
    readonly: bool,
}

impl BuildAsmExpand {
    pub fn new(asm: String) -> Self {
        BuildAsmExpand {
            asm,
            ..Default::default()
        }
    }

    // Takes by reference because of syntax reasons, since normal Rust allows immutables that are
    // uninitialized and only assigned once (i.e. as the output for an assembly macro).
    // We also only assign once, so reference gives the correct semantics.
    pub fn push_output<T: HasValue>(mut self, scope: &Scope, output: &T) -> Self {
        let value = output.value(scope);
        self.out_values.push(value);
        self
    }

    pub fn push_input<T: HasValue>(mut self, scope: &Scope, input: T) -> Self {
        let value = input.value(scope);
        self.in_values.push(value);
        self
    }

    pub fn pure(mut self) -> Self {
        self.pure = true;
        self
    }

    pub fn nomem(mut self) -> Self {
        self.nomem = true;
        self
    }

    pub fn readonly(mut self) -> Self {
        self.readonly = true;
        self
    }

    pub fn register(self, scope: &Scope) {
        let ctx = scope.ctx_mut();
        let result_types = self
            .out_values
            .iter()
            .map(|it| it.get_type(ctx).as_ptr(ctx).inner)
            .collect();
        let op = InlineAsmOp::new(ctx, result_types, self.asm, self.in_values);
        if self.pure {
            op.set_pure(ctx);
        }
        if self.nomem {
            op.set_nomem(ctx);
        }
        if self.readonly {
            op.set_readonly(ctx);
        }
        scope.register(&op);
        // Store results back to out expand values
        for (&out_ptr, result) in self.out_values.iter().zip(op.get_operation().results(ctx)) {
            assign::expand_element(scope, result.into(), out_ptr.into());
        }
    }
}
