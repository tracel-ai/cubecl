use cubecl_core::{
    self as cubecl,
    ir::{
        CoopMma, Instruction, MatrixIdent, Operation, Processor, Scope, ScopeProcessing, Variable,
    },
    prelude::*,
};

#[derive(new, Debug)]
pub struct HipMmaProcessor;

impl Processor for HipMmaProcessor {
    fn transform(&self, mut processing: ScopeProcessing) -> ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            match instruction.operation {
                Operation::CoopMma(CoopMma::RowIndex { lane_id, i, matrix }) => {
                    let scope =
                        Scope::root(false).with_global_state(processing.global_state.clone());
                    let row_idx: Variable =
                        row_index::expand(&scope, lane_id.into(), i.into(), matrix.ident).into();
                    let tmp_processing = scope.process([]);
                    for inst in tmp_processing.instructions {
                        processing.instructions.push(inst);
                    }
                    for var in tmp_processing.variables {
                        processing.variables.push(var);
                    }

                    processing.instructions.push(Instruction::new(
                        Operation::Copy(row_idx),
                        instruction.out(),
                    ));
                }
                Operation::CoopMma(CoopMma::ColIndex { lane_id, i, matrix }) => {
                    let scope =
                        Scope::root(false).with_global_state(processing.global_state.clone());
                    let row_idx: Variable =
                        col_index::expand(&scope, lane_id.into(), i.into(), matrix.ident).into();
                    let tmp_processing = scope.process([]);
                    for inst in tmp_processing.instructions {
                        processing.instructions.push(inst);
                    }
                    for var in tmp_processing.variables {
                        processing.variables.push(var);
                    }

                    processing.instructions.push(Instruction::new(
                        Operation::Copy(row_idx),
                        instruction.out(),
                    ));
                }
                _ => {
                    processing.instructions.push(instruction);
                }
            }
        }

        processing
    }
}

#[cube]
fn row_index(lane_id: u32, i: u32, #[comptime] ident: MatrixIdent) -> u32 {
    match ident {
        MatrixIdent::A => lane_id % 16,
        MatrixIdent::B => i,
        // 2 * i, offset by 1 if lane_id >= 16
        MatrixIdent::Accumulator => i * 2 + (lane_id / 16),
    }
}

#[cube]
fn col_index(lane_id: u32, i: u32, #[comptime] ident: MatrixIdent) -> u32 {
    match ident {
        MatrixIdent::A => i,
        MatrixIdent::B => lane_id % 16,
        MatrixIdent::Accumulator => lane_id % 16,
    }
}
