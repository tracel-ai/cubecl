use cubecl_ir::prelude::*;
use pliron::{common_traits::Named, identifier::Identifier};

pub trait WgslValue {
    fn name(&self, ctx: &Context) -> Identifier;
    fn fmt_left(&self, ctx: &Context) -> String;
}

impl WgslValue for Value {
    fn name(&self, ctx: &Context) -> Identifier {
        sanitize_name(self.unique_name(ctx))
    }

    fn fmt_left(&self, ctx: &Context) -> String {
        let name = self.name(ctx);
        format!("let {}", name)
    }
}

fn sanitize_name(name: Identifier) -> Identifier {
    // WGSL reserves `__` as a prefix, so any variable starting with that would be a compile error
    if name.as_ref().starts_with("__") {
        return Identifier::try_new(format!("v{}", name.as_ref())).unwrap();
    }
    name
}
