use alloc::{string::String, vec::Vec};

use cubecl_ir::{
    AddressSpace, Scope,
    dialect::{InlineAsmOp, MemoryClobbers, OperationPtrExt},
    interfaces::TypeExt,
};
use pliron::{op::Op, r#type::Typed, value::Value};

use crate::frontend::{HasValue, assign};

pub use cubecl_ir::dialect::{InputKind, InputSpec, RegSpec};

#[derive(Default)]
pub struct BuildAsmExpand {
    asm: String,
    out_values: Vec<Value>,
    out_specs: Vec<RegSpec>,
    in_values: Vec<Value>,
    in_specs: Vec<InputSpec>,
    pure: bool,
    nomem: bool,
    explicit_mem: bool,
    readonly: bool,
    reads_spaces: Vec<AddressSpace>,
    writes_spaces: Vec<AddressSpace>,
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
    pub fn push_output<T: HasValue>(mut self, scope: &Scope, output: &T, spec: RegSpec) -> Self {
        let value = output.value(scope);
        self.out_values.push(value);
        self.out_specs.push(spec);
        self
    }

    pub fn push_input<T: HasValue>(
        mut self,
        scope: &Scope,
        input: T,
        kind: InputKind,
        spec: RegSpec,
    ) -> Self {
        let value = input.value(scope);
        self.in_values.push(value);
        self.in_specs.push(InputSpec::new(kind, spec));
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

    pub fn explicit_mem(mut self) -> Self {
        self.explicit_mem = true;
        self
    }

    pub fn readonly(mut self) -> Self {
        self.readonly = true;
        self
    }

    pub fn reads_local(mut self) -> Self {
        self.reads_spaces.push(AddressSpace::Local);
        self
    }

    pub fn reads_shared(mut self) -> Self {
        self.reads_spaces.push(AddressSpace::Shared);
        self
    }

    pub fn reads_global(mut self) -> Self {
        self.reads_spaces.push(AddressSpace::Global(0));
        self
    }

    pub fn writes_local(mut self) -> Self {
        self.writes_spaces.push(AddressSpace::Local);
        self
    }

    pub fn writes_shared(mut self) -> Self {
        self.writes_spaces.push(AddressSpace::Shared);
        self
    }

    pub fn writes_global(mut self) -> Self {
        self.writes_spaces.push(AddressSpace::Global(0));
        self
    }

    pub fn register(self, scope: &Scope) {
        let ctx = scope.ctx_mut();
        let result_types = self
            .out_values
            .iter()
            .map(|it| it.get_type(ctx).as_ptr(ctx).inner)
            .collect();
        let memory_clobbers = if self.nomem {
            MemoryClobbers::Nomem
        } else if self.readonly {
            MemoryClobbers::Readonly
        } else if self.explicit_mem {
            MemoryClobbers::Explicit {
                reads_spaces: self.reads_spaces.into(),
                writes_spaces: self.writes_spaces.into(),
            }
        } else {
            MemoryClobbers::ReadWrite
        };
        let op = InlineAsmOp::new(
            ctx,
            result_types,
            self.out_specs,
            self.asm,
            memory_clobbers,
            self.in_values,
            self.in_specs,
        );
        if self.pure {
            op.set_pure(ctx);
        }
        scope.register(&op);
        // Store results back to out expand values
        for (&out_ptr, result) in self.out_values.iter().zip(op.get_operation().results(ctx)) {
            assign::expand_element(scope, result.into(), out_ptr.into());
        }
    }
}
