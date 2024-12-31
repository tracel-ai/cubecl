pub mod cube_comment {
    use crate::{ir::NonSemantic, prelude::CubeContext};

    pub fn expand(context: &mut CubeContext, content: &str) {
        context.register(NonSemantic::Comment {
            content: content.to_string(),
        });
    }
}
