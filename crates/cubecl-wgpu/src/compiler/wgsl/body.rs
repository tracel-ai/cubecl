use crate::compiler::wgsl::Elem;

use super::Instruction;
use std::fmt::Display;

/// A body is composed of a list of [instructions](Instruction).
///
/// Note that the body assumes that the kernel will run on a 2D grid defined by the workgroup size
/// X and Y, but with Z=1.
#[derive(Debug, Clone)]
pub struct Body {
    pub instructions: Vec<Instruction>,
    pub id: bool,
    pub address_type: Elem,
}

impl Display for Body {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.id {
            let addr_ty = self.address_type;
            writeln!(
                f,
                "let id = ({addr_ty}(global_id.z) * {addr_ty}(num_workgroups.x) * {addr_ty}(WORKGROUP_SIZE_X) * {addr_ty}(num_workgroups.y) * {addr_ty}(WORKGROUP_SIZE_Y)) + ({addr_ty}(global_id.y) * {addr_ty}(num_workgroups.x) * {addr_ty}(WORKGROUP_SIZE_X)) + {addr_ty}(global_id.x);\n",
            )?;
        }

        for ops in self.instructions.iter() {
            write!(f, "{ops}")?;
        }

        Ok(())
    }
}
