pub mod cube_comment {
    use crate::{ir::Comment, prelude::CubeContext};

    pub fn expand(context: &mut CubeContext, content: &str) {
        context.register(Comment {
            content: content.to_string(),
        })
    }
}
