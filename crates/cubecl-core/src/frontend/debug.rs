use crate::ir::DebugInfo;

use super::CubeContext;

pub fn debug_call_expand<C>(
    context: &mut CubeContext,
    name: &'static str,
    call: impl FnOnce(&mut CubeContext) -> C,
) -> C {
    if context.debug_enabled {
        context.register(DebugInfo::BeginCall {
            name: name.to_string(),
        });

        let ret = call(context);

        context.register(DebugInfo::EndCall);

        ret
    } else {
        call(context)
    }
}
