use core::{
    fmt,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign},
};

use alloc::string::ToString;
use derive_more::From;
use itertools::Itertools;
use pliron::{
    opts::dce::SideEffects,
    printable::{self, Printable},
};

use crate::{EnumSet, EnumSetType, dialect::OperationExt, prelude::*, typed_vec_attr};

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

/// Memory scope for effects, they typically affect *all* global buffers and often affect more than
/// one storage class. This enum set allows properly specifying these effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumSetType)]
#[format]
pub enum MemoryScope {
    Global,
    Shared,
    Local,
}

typed_vec_attr!(MemoryScope, "cube.memory_scopes", MemoryScopeVecAttr);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extent {
    // Exactly this pointer, without requiring a size
    ExactPointer,
    Exact(usize),
    AtMost(usize),
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, From)]
pub enum MemoryLocation {
    Pointer(Value, Extent),
    Scope(EnumSet<MemoryScope>),
    All,
}

impl MemoryLocation {
    pub fn exact_pointer(value: Value) -> MemoryLocation {
        MemoryLocation::Pointer(value, Extent::ExactPointer)
    }
}

impl From<MemoryScope> for MemoryLocation {
    fn from(value: MemoryScope) -> Self {
        MemoryLocation::Scope(value.into())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryEffect {
    Read(MemoryLocation),
    Write(MemoryLocation),
    /// Make a memory location available, which references it so stores can't be sunk past this
    MakeAvailable(MemoryLocation),
    /// Make writes by other threads to a memory location visible to the current thread
    MakeVisible(MemoryLocation),
    /// Not analyzable, clobber the entire state
    Opaque,
}

impl MemoryEffect {
    pub fn location(&self) -> MemoryLocation {
        match self {
            MemoryEffect::Read(loc) | MemoryEffect::Write(loc) => *loc,
            MemoryEffect::MakeVisible(loc) | MemoryEffect::MakeAvailable(loc) => *loc,
            MemoryEffect::Opaque => MemoryLocation::All,
        }
    }

    pub fn pointer(&self) -> Option<(Value, Extent)> {
        match self.location() {
            MemoryLocation::Pointer(value, extent) => Some((value, extent)),
            MemoryLocation::Scope(_) | MemoryLocation::All => None,
        }
    }

    pub fn mod_ref_mask(&self) -> ModRefResult {
        match self {
            MemoryEffect::Read(_) | MemoryEffect::MakeAvailable(_) => ModRefResult::Ref,
            MemoryEffect::Write(_) | MemoryEffect::MakeVisible(_) => ModRefResult::Mod,
            MemoryEffect::Opaque => ModRefResult::ModRef,
        }
    }

    pub fn modifies(&self) -> bool {
        self.mod_ref_mask().contains_mod()
    }
}

impl Printable for Extent {
    fn fmt(&self, _: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Extent::ExactPointer => f.write_str("ExactPointer"),
            Extent::Exact(value) => write!(f, "Exact({value})"),
            Extent::AtMost(value) => write!(f, "AtMost({value})"),
            Extent::Unknown => write!(f, "Unknown"),
        }
    }
}

impl Printable for MemoryLocation {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryLocation::Pointer(value, extent) => {
                write!(
                    f,
                    "Pointer({}, extent: {})",
                    value.disp(ctx),
                    extent.disp(ctx)
                )
            }
            MemoryLocation::Scope(scope) => write!(
                f,
                "Scope({})",
                scope.iter().map(|it| it.disp(ctx).to_string()).join(" | ")
            ),
            MemoryLocation::All => f.write_str("All"),
        }
    }
}

impl Printable for MemoryEffect {
    fn fmt(&self, ctx: &Context, _: &printable::State, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryEffect::Read(loc) => write!(f, "Read({})", loc.disp(ctx)),
            MemoryEffect::Write(loc) => write!(f, "Write({})", loc.disp(ctx)),
            MemoryEffect::MakeAvailable(loc) => write!(f, "MakeAvailable({})", loc.disp(ctx)),
            MemoryEffect::MakeVisible(loc) => write!(f, "MakeVisible({})", loc.disp(ctx)),
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
