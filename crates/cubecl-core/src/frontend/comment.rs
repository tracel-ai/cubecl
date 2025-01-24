pub mod cube_comment {
    use crate::ir::NonSemantic;
    use cubecl_ir::Scope;

    pub fn expand(context: &mut Scope, content: &str) {
        context.register(NonSemantic::Comment {
            content: content.to_string(),
        });
    }
}
