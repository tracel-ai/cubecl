//! Device-capability validation of compiled SPIR-V modules.

use std::collections::HashMap;

use cubecl_ir::{DeviceProperties, ElemType, FloatKind, IntKind, UIntKind};
use rspirv::{
    dr::{Instruction, Module, Operand},
    spirv::{CooperativeMatrixUse, FPEncoding, Op, Scope},
};

/// Checks that every subgroup-scope cooperative matrix type declared in the
/// module matches a fragment shape the device advertises. Drivers only accept
/// the exact (element type, rows, columns, use) combinations they report;
/// anything else is undefined behavior at runtime — loads and stores silently
/// produce garbage on at least RADV — so it must be rejected at compile time.
pub fn check_cmma_types(module: &Module, props: &DeviceProperties) -> Result<(), String> {
    let types: HashMap<u32, &Instruction> = module
        .types_global_values
        .iter()
        .filter_map(|inst| inst.result_id.map(|id| (id, inst)))
        .collect();

    let const_u32 = |id: &u32| -> Option<u32> {
        let inst = types.get(id)?;
        if inst.class.opcode != Op::Constant {
            return None;
        }
        match inst.operands.first()? {
            Operand::LiteralBit32(v) => Some(*v),
            _ => None,
        }
    };

    for inst in &module.types_global_values {
        if inst.class.opcode != Op::TypeCooperativeMatrixKHR {
            continue;
        }
        let [
            Operand::IdRef(component),
            Operand::IdRef(scope) | Operand::IdScope(scope),
            Operand::IdRef(rows),
            Operand::IdRef(cols),
            Operand::IdRef(usage),
        ] = inst.operands.as_slice()
        else {
            continue;
        };
        // Workgroup-scope matrices (cooperative matrix 2) have flexible
        // dimensions and are validated by their own feature set.
        if const_u32(scope) != Some(Scope::Subgroup as u32) {
            continue;
        }
        let (Some(rows), Some(cols), Some(usage)) =
            (const_u32(rows), const_u32(cols), const_u32(usage))
        else {
            continue;
        };
        let Some(elem) = types.get(component).and_then(|it| elem_type(it)) else {
            continue;
        };
        let Some(usage) = CooperativeMatrixUse::from_u32(usage) else {
            continue;
        };

        let supported = props.features.matmul.cmma.iter().any(|cfg| match usage {
            CooperativeMatrixUse::MatrixAKHR => cfg.a_type == elem && cfg.m == rows && cfg.k == cols,
            CooperativeMatrixUse::MatrixBKHR => cfg.b_type == elem && cfg.k == rows && cfg.n == cols,
            CooperativeMatrixUse::MatrixAccumulatorKHR => {
                cfg.cd_type == elem && cfg.m == rows && cfg.n == cols
            }
        });
        if !supported {
            let usage = match usage {
                CooperativeMatrixUse::MatrixAKHR => "A",
                CooperativeMatrixUse::MatrixBKHR => "B",
                CooperativeMatrixUse::MatrixAccumulatorKHR => "Accumulator",
            };
            return Err(format!(
                "the device doesn't support a {rows}x{cols} {usage} cooperative matrix fragment \
                 of {elem:?}; supported configurations: {:?}",
                props.features.matmul.cmma
            ));
        }
    }

    Ok(())
}

fn elem_type(inst: &Instruction) -> Option<ElemType> {
    let width = match inst.operands.first()? {
        Operand::LiteralBit32(w) => *w,
        _ => return None,
    };
    let ty = match inst.class.opcode {
        Op::TypeFloat => match inst.operands.get(1) {
            None => match width {
                16 => ElemType::Float(FloatKind::F16),
                32 => ElemType::Float(FloatKind::F32),
                64 => ElemType::Float(FloatKind::F64),
                _ => return None,
            },
            Some(Operand::FPEncoding(enc)) => match enc {
                FPEncoding::BFloat16KHR => ElemType::Float(FloatKind::BF16),
                FPEncoding::Float8E4M3EXT => ElemType::Float(FloatKind::E4M3),
                FPEncoding::Float8E5M2EXT => ElemType::Float(FloatKind::E5M2),
                _ => return None,
            },
            Some(_) => return None,
        },
        Op::TypeInt => {
            let signed = matches!(inst.operands.get(1)?, Operand::LiteralBit32(1));
            match (width, signed) {
                (8, true) => ElemType::Int(IntKind::I8),
                (16, true) => ElemType::Int(IntKind::I16),
                (32, true) => ElemType::Int(IntKind::I32),
                (64, true) => ElemType::Int(IntKind::I64),
                (8, false) => ElemType::UInt(UIntKind::U8),
                (16, false) => ElemType::UInt(UIntKind::U16),
                (32, false) => ElemType::UInt(UIntKind::U32),
                (64, false) => ElemType::UInt(UIntKind::U64),
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(ty)
}
