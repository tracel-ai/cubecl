use cubecl_core::{
    self as cubecl,
    ir::{
        CoopMma, Instruction, MatrixIdent, Operation, Processor, Scope, ScopeProcessing, Variable,
    },
    prelude::*,
};

#[derive(new, Debug)]
pub struct CudaMmaProcessor;

impl Processor for CudaMmaProcessor {
    fn transform(&self, mut processing: ScopeProcessing) -> ScopeProcessing {
        let mut instructions = Vec::new();
        core::mem::swap(&mut processing.instructions, &mut instructions);

        for instruction in instructions {
            match instruction.operation {
                Operation::CoopMma(CoopMma::RowIndex { lane_id, i, matrix }) => {
                    let elems_per_reg = 32 / matrix.storage.elem_type().size_bits();
                    let scope =
                        Scope::root(false).with_global_state(processing.global_state.clone());
                    let row_idx: Variable = row_index::expand(
                        &scope,
                        lane_id.into(),
                        i.into(),
                        elems_per_reg as u32,
                        matrix.ident,
                    )
                    .into();
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
                    let elems_per_reg = 32 / matrix.storage.elem_type().size_bits();
                    let scope =
                        Scope::root(false).with_global_state(processing.global_state.clone());
                    let col_idx: Variable = col_index::expand(
                        &scope,
                        lane_id.into(),
                        i.into(),
                        elems_per_reg as u32,
                        matrix.ident,
                    )
                    .into();
                    let tmp_processing = scope.process([]);
                    for inst in tmp_processing.instructions {
                        processing.instructions.push(inst);
                    }
                    for var in tmp_processing.variables {
                        processing.variables.push(var);
                    }

                    processing.instructions.push(Instruction::new(
                        Operation::Copy(col_idx),
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

/// Derived from PTX shape documentation
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma>
#[cube]
fn row_index(
    lane_id: u32,
    i: u32,
    #[comptime] elems_per_reg: u32,
    #[comptime] ident: MatrixIdent,
) -> u32 {
    match ident {
        MatrixIdent::A => {
            let group_id = lane_id / 4;
            let odd_register = (i / elems_per_reg) & 1;
            group_id + odd_register * 8
        }
        MatrixIdent::B => {
            let thread_id_in_group = lane_id % 4;
            let offset = thread_id_in_group * elems_per_reg + (i % elems_per_reg);
            let reg = i / elems_per_reg;
            offset + elems_per_reg * 4 * reg
        }
        MatrixIdent::Accumulator => {
            let group_id = lane_id / 4;
            let offset = (i << 2) & 8;
            group_id + offset
        }
    }
}

/// Derived from PTX shape documentation
/// <https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-for-mma>
#[cube]
fn col_index(
    lane_id: u32,
    i: u32,
    #[comptime] elems_per_reg: u32,
    #[comptime] ident: MatrixIdent,
) -> u32 {
    match ident {
        MatrixIdent::A => {
            let thread_id_in_group = lane_id % 4;
            let offset = thread_id_in_group * elems_per_reg + (i % elems_per_reg);
            let group_2 = (i / (2 * elems_per_reg)) & 1;
            offset + 4 * elems_per_reg * group_2
        }
        MatrixIdent::B => lane_id >> 2,
        MatrixIdent::Accumulator => {
            let thread_id_in_group = lane_id % 4;
            (thread_id_in_group * 2) + (i % 2)
        }
    }
}
