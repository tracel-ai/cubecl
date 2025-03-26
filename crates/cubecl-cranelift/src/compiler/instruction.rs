use cranelift::prelude::{
    types, Block, EntityRef, FunctionBuilder, InstBuilder, Type, Value,
    Variable as CraneliftVariable,
};
use cubecl_core::{
    ir::{Arithmetic, Instruction as CubeInstruction, Variable as CubeVariable},
    prelude::KernelDefinition,
};

use super::{to_ssa_type, CompilerState};

impl<'a> CompilerState<'a> {
    pub(crate) fn process_scope(&mut self, kernel: &KernelDefinition) {
        let scope_process = kernel.body.clone().process();

        scope_process.variables.iter().for_each(|var| {
            let cl_var = self.lookup.getsert_var(var.kind);
            self.func_builder
                .declare_var(cl_var, to_ssa_type(&var.item));
        });
        scope_process.operations.iter().for_each(|op| {
            self.translate_instruction(op);
        });
    }
    fn translate_instruction(&mut self, op: &CubeInstruction) {
        self.get_output_val(op);
    }

    fn get_output_val(&mut self, op: &CubeInstruction) -> Option<Value> {
        match &op.operation {
            cubecl_core::ir::Operation::Copy(variable) => {
                if let (Some(output), Some(input)) = (&op.out, self.lookup.get(variable.kind)) {
                    //dlet output = self.lookup.get(output.kind).unwrap();
                    Some(self.func_builder.use_var(*input))
                } else {
                    None
                }
            }
            cubecl_core::ir::Operation::Arithmetic(arithmetic) => match arithmetic {
                Arithmetic::Fma(fma_operator) => {
                    let (x, y, z) = (
                        self.get_var(&fma_operator.a),
                        self.get_var(&fma_operator.b),
                        self.get_var(&fma_operator.c),
                    );
                    let fma = self.func_builder.ins().fma(x, y, z);
                    Some(fma)
                }
                Arithmetic::Add(binary_operator) => match self.bin_args(&op.out, binary_operator) {
                    BinOpKind::Int(bin_op) | BinOpKind::Uint(bin_op) => {
                        Some(self.func_builder.ins().iadd(bin_op.lhs, bin_op.rhs))
                    }

                    BinOpKind::Float(bin_op) => {
                        Some(self.func_builder.ins().fadd(bin_op.lhs, bin_op.rhs))
                    }
                },
                Arithmetic::Sub(binary_operator) => match self.bin_args(&op.out, binary_operator) {
                    BinOpKind::Int(bin_op) | BinOpKind::Uint(bin_op) => {
                        Some(self.func_builder.ins().isub(bin_op.lhs, bin_op.rhs))
                    }

                    BinOpKind::Float(bin_op) => {
                        Some(self.func_builder.ins().fsub(bin_op.lhs, bin_op.rhs))
                    }
                },
                Arithmetic::Mul(binary_operator) => match self.bin_args(&op.out, binary_operator) {
                    BinOpKind::Int(bin_op) | BinOpKind::Uint(bin_op) => {
                        Some(self.func_builder.ins().imul(bin_op.lhs, bin_op.rhs))
                    }

                    BinOpKind::Float(bin_op) => {
                        Some(self.func_builder.ins().fmul(bin_op.lhs, bin_op.rhs))
                    }
                },
                Arithmetic::Div(binary_operator) => match self.bin_args(&op.out, binary_operator) {
                    BinOpKind::Int(_bin_op) | BinOpKind::Uint(_bin_op) => {
                        //.Some(self.func_builder.ins().idiv(bin_op.lhs, bin_op.rhs))
                        todo!()
                    }

                    BinOpKind::Float(bin_op) => {
                        Some(self.func_builder.ins().fdiv(bin_op.lhs, bin_op.rhs))
                    }
                },
                Arithmetic::Powf(binary_operator) => {
                    match self.bin_args(&op.out, binary_operator) {
                        BinOpKind::Int(bin_op) | BinOpKind::Uint(bin_op) => {
                            //Some(self.func_builder.ins().imul(bin_op.lhs, bin_op.rhs))
                            todo!()
                        }

                        BinOpKind::Float(bin_op) => {
                            Some(self.func_builder.ins().fmul(bin_op.lhs, bin_op.rhs))
                        }
                    }
                }
                Arithmetic::Modulo(binary_operator) => todo!(),
                Arithmetic::Max(binary_operator) => todo!(),
                Arithmetic::Min(binary_operator) => todo!(),
                Arithmetic::Remainder(binary_operator) => todo!(),
                Arithmetic::Dot(binary_operator) => todo!(),
                Arithmetic::Abs(unary_operator) => todo!(),
                Arithmetic::Exp(unary_operator) => todo!(),
                Arithmetic::Log(unary_operator) => todo!(),
                Arithmetic::Log1p(unary_operator) => todo!(),
                Arithmetic::Cos(unary_operator) => todo!(),
                Arithmetic::Sin(unary_operator) => todo!(),
                Arithmetic::Tanh(unary_operator) => todo!(),

                Arithmetic::Sqrt(unary_operator) => todo!(),
                Arithmetic::Round(unary_operator) => todo!(),
                Arithmetic::Floor(unary_operator) => todo!(),
                Arithmetic::Ceil(unary_operator) => todo!(),
                Arithmetic::Erf(unary_operator) => todo!(),
                Arithmetic::Recip(unary_operator) => todo!(),
                Arithmetic::Clamp(clamp_operator) => todo!(),

                Arithmetic::Neg(unary_operator) => todo!(),

                Arithmetic::Magnitude(unary_operator) => todo!(),
                Arithmetic::Normalize(unary_operator) => todo!(),
            },
            cubecl_core::ir::Operation::Comparison(comparison) => todo!(),
            cubecl_core::ir::Operation::Bitwise(bitwise) => todo!(),
            cubecl_core::ir::Operation::Operator(operator) => todo!(),
            cubecl_core::ir::Operation::Atomic(atomic_op) => todo!(),
            cubecl_core::ir::Operation::Metadata(metadata) => todo!(),
            cubecl_core::ir::Operation::Branch(branch) => todo!(),
            cubecl_core::ir::Operation::Synchronization(synchronization) => todo!(),
            cubecl_core::ir::Operation::Plane(plane) => todo!(),
            cubecl_core::ir::Operation::CoopMma(coop_mma) => todo!(),
            cubecl_core::ir::Operation::Pipeline(pipeline_ops) => todo!(),
            cubecl_core::ir::Operation::NonSemantic(non_semantic) => todo!(),
        }
    }
    fn bin_args(
        &mut self,
        out: &Option<CubeVariable>,
        binary_operator: &cubecl_core::ir::BinaryOperator,
    ) -> BinOpKind {
        let lhs = self.get_var(&binary_operator.lhs);
        let rhs = self.lookup.get(binary_operator.rhs.kind).unwrap().clone();
        let rhs = self.func_builder.use_var(rhs);
        let out = out.expect("Binary op requires output");
        BinOpKind::new(BinOp { lhs, rhs, out })
    }

    fn get_var(&mut self, var: &CubeVariable) -> Value {
        self.func_builder
            .use_var(self.lookup.get(var.kind).unwrap().clone())
    }

    fn bin_op(
        &mut self,
        binary_operator: &cubecl_core::ir::BinaryOperator,
        out: Option<CubeVariable>,
        mut f: impl FnMut(Value, Value) -> Value,
    ) {
        let lhs = self.lookup.get(binary_operator.lhs.kind).unwrap().clone();
        let rhs = self.lookup.get(binary_operator.rhs.kind).unwrap().clone();
        let lhs = self.func_builder.use_var(lhs);
        let rhs = self.func_builder.use_var(rhs);

        let res = f(lhs, rhs);

        if out.is_some() {
            let out = self.lookup.get(out.unwrap().kind).unwrap().clone();
            self.func_builder.def_var(out, res);
        }
    }
}
struct BinOp {
    lhs: Value,
    rhs: Value,
    out: CubeVariable,
}
enum BinOpKind {
    Int(BinOp),
    Uint(BinOp),
    Float(BinOp),
}

impl BinOpKind {
    fn new(op: BinOp) -> BinOpKind {
        match op.out.elem() {
            cubecl_core::ir::Elem::Float(float_kind) => BinOpKind::Float(op),
            cubecl_core::ir::Elem::Int(int_kind) => BinOpKind::Int(op),
            cubecl_core::ir::Elem::UInt(uint_kind) => BinOpKind::Uint(op),
            // cubecl_core::ir::Elem::AtomicFloat(float_kind) => todo!(),
            // cubecl_core::ir::Elem::AtomicInt(int_kind) => todo!(),
            // cubecl_core::ir::Elem::AtomicUInt(uint_kind) => todo!(),
            _ => {
                panic!("Unsupported output type for binary op")
            }
        }
    }
}
