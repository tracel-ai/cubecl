use cubecl_ir::prelude::*;
use pliron::{common_traits::Named, identifier::Identifier};

pub trait WgslValue {
    fn name(&self, ctx: &Context) -> Identifier;
    fn fmt_left(&self, ctx: &Context) -> String;
}

impl WgslValue for Value {
    fn name(&self, ctx: &Context) -> Identifier {
        self.unique_name(ctx)
    }

    fn fmt_left(&self, ctx: &Context) -> String {
        let name = self.name(ctx);
        format!("let {}", name)
    }
}
