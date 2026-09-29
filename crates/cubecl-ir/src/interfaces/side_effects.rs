use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

use pliron::{builtin::ops::ConstantOp, opts::dce::SideEffects, printable::Printable};

use crate::{AddressSpace, NoMemoryEffect, dialect::OperationExt, prelude::*};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum ModRefResult {
    NoModRef,
    Ref,
    Mod,
    ModRef,
}

impl ModRefResult {
    pub fn join(&self, other: ModRefResult) -> ModRefResult {
        match (*self, other) {
            (ModRefResult::NoModRef, other) | (other, ModRefResult::NoModRef) => other,
            (_, ModRefResult::ModRef) | (ModRefResult::ModRef, _) => ModRefResult::ModRef,
            (ModRefResult::Ref, ModRefResult::Ref) => ModRefResult::Ref,
            (ModRefResult::Mod, ModRefResult::Mod) => ModRefResult::Mod,
            (ModRefResult::Ref, ModRefResult::Mod) | (ModRefResult::Mod, ModRefResult::Ref) => {
                ModRefResult::ModRef
            }
        }
    }

    pub fn meet(&self, other: ModRefResult) -> ModRefResult {
        match (*self, other) {
            (ModRefResult::NoModRef, _) | (_, ModRefResult::NoModRef) => ModRefResult::NoModRef,
            (ModRefResult::ModRef, other) | (other, ModRefResult::ModRef) => other,
            (ModRefResult::Ref, ModRefResult::Ref) => ModRefResult::Ref,
            (ModRefResult::Mod, ModRefResult::Mod) => ModRefResult::Mod,
            (ModRefResult::Mod, ModRefResult::Ref) | (ModRefResult::Ref, ModRefResult::Mod) => {
                ModRefResult::NoModRef
            }
        }
    }

    pub fn contains_mod(&self) -> bool {
        match self {
            ModRefResult::NoModRef | ModRefResult::Ref => false,
            ModRefResult::Mod | ModRefResult::ModRef => true,
        }
    }
}

impl BitOr for ModRefResult {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self::Output {
        self.join(rhs)
    }
}

impl BitAnd for ModRefResult {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self::Output {
        self.meet(rhs)
    }
}

impl BitOrAssign for ModRefResult {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.join(rhs);
    }
}

impl BitAndAssign for ModRefResult {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = self.meet(rhs);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryEffect {
    Read(Value),
    Write(Value),
    ReadAllInSpace(AddressSpace),
    WriteAllInSpace(AddressSpace),
    ReadAll,
    WriteAll,
    // Not analyzable, clobber the entire state
    Opaque,
}

impl MemoryEffect {
    pub fn value(&self) -> Option<Value> {
        match self {
            MemoryEffect::Read(value) | MemoryEffect::Write(value) => Some(*value),
            MemoryEffect::ReadAllInSpace(_)
            | MemoryEffect::WriteAllInSpace(_)
            | MemoryEffect::ReadAll
            | MemoryEffect::WriteAll
            | MemoryEffect::Opaque => None,
        }
    }

    pub fn mod_ref_mask(&self) -> ModRefResult {
        match self {
            MemoryEffect::Read(_) | MemoryEffect::ReadAllInSpace(_) | MemoryEffect::ReadAll => {
                ModRefResult::Ref
            }
            MemoryEffect::Write(_) | MemoryEffect::WriteAllInSpace(_) | MemoryEffect::WriteAll => {
                ModRefResult::Mod
            }
            MemoryEffect::Opaque => ModRefResult::ModRef,
        }
    }

    pub fn modifies(&self) -> bool {
        self.mod_ref_mask().contains_mod()
    }
}

impl Printable for MemoryEffect {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match self {
            MemoryEffect::Read(value) => write!(f, "Read({})", value.disp(ctx)),
            MemoryEffect::Write(value) => write!(f, "Write({})", value.disp(ctx)),
            MemoryEffect::ReadAllInSpace(address_space) => {
                write!(f, "ReadAllInSpace({})", address_space.disp(ctx))
            }
            MemoryEffect::WriteAllInSpace(address_space) => {
                write!(f, "WriteAllInSpace({})", address_space.disp(ctx))
            }
            MemoryEffect::ReadAll => write!(f, "ReadAll"),
            MemoryEffect::WriteAll => write!(f, "WriteAll"),
            MemoryEffect::Opaque => write!(f, "Opaque"),
        }
    }
}

#[op_interface]
pub trait MemoryEffectsOp {
    verify_op_succ!();
    fn memory_effects(&self, ctx: &Context) -> Vec<MemoryEffect>;
    fn has_effects(&self, ctx: &Context) -> bool {
        !self.memory_effects(ctx).is_empty()
    }
}

NoMemoryEffect!(ConstantOp);

pub fn has_memory_side_effects(effects: &[MemoryEffect]) -> bool {
    effects.iter().any(|effect| effect.modifies())
}

pub fn get_nested_side_effects(ctx: &Context, op: Ptr<Operation>) -> bool {
    op.deref(ctx).immediately_nested_ops(ctx).any(|op| {
        op_cast::<dyn SideEffects>(&*op.dyn_op(ctx)).is_none_or(|op| op.has_side_effects(ctx))
    })
}

pub fn get_nested_memory_effects(ctx: &Context, op: Ptr<Operation>) -> Vec<MemoryEffect> {
    let mut out = vec![];
    for op in op.deref(ctx).immediately_nested_ops(ctx) {
        out.extend(match op_cast::<dyn MemoryEffectsOp>(&*op.dyn_op(ctx)) {
            Some(mem_op) => mem_op.memory_effects(ctx),
            None => vec![MemoryEffect::Opaque],
        });
    }
    out
}

pub enum Speculatability {
    /// The Operation in question cannot be speculatively executed. This could be
    /// because it may invoke undefined behavior or have other side effects.
    NotSpeculatable,

    // The Operation in question can be speculatively executed. It does not have
    // any side effects or undefined behavior.
    Speculatable,

    // The Operation in question can be speculatively executed if all the
    // operations in all attached regions can also be speculatively executed.
    RecursivelySpeculatable,
}

#[op_interface]
pub trait ConditionallySpeculatable {
    verify_op_succ!();
    fn speculatability(&self, ctx: &Context) -> Speculatability;
}
