use cubecl_ir::{dialect::cmp::EqualOp, prelude::*};

use crate::ops::to_spirv_dialect::ToSpirvDialectOp;

// impl ToSpirvDialectOp for EqualOp {
//     fn to_spirv_dialect(
//         &self,
//         ctx: &mut Context,
//         rewriter: &mut DialectConversionRewriter,
//         operands_info: &OperandsInfo,
//     ) -> Result<()> {
//         let lhs = self.get_operation().operand(ctx, 0);
//         let rhs = self.get_operation().operand(ctx, 1);

//         Ok(())
//     }
// }
