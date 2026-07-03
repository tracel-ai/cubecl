use hashbrown::HashMap;
use internment::Intern;

use crate::{
    AddressSpace, Arithmetic, Branch, ElemType, Memory, Operation, OperationReflect, Type,
    UIntKind, Value,
};

use alloc::{format, string::String, vec::Vec};

/// The unit a counted operation is measured in.
///
/// Part of [`CountKey`], so op counts and byte counts never share a slot even if two ops happened
/// to share a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CountUnit {
    /// A number of scalar operations.
    Ops,
    /// A number of bytes moved.
    Bytes,
}

impl CountUnit {
    /// Short suffix used when displaying counts (e.g. `"ops"`, `"bytes"`).
    pub fn suffix(self) -> &'static str {
        match self {
            CountUnit::Ops => "ops",
            CountUnit::Bytes => "bytes",
        }
    }
}

/// The identity a count accumulates under: the operation name, the element type it acts on, and
/// the unit it is measured in.
///
/// `name` is interned, so equality/hashing are pointer-based — a collision-free, hash-like
/// identity — while the readable name is kept for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CountKey {
    /// Interned operation name, e.g. `"Add"` or `"Load"`.
    pub name: Intern<str>,
    /// The element type the operation acts on.
    pub elem: ElemType,
    /// What the operation is measured in.
    pub unit: CountUnit,
}

/// One operation's contribution to the counts: what to accumulate, and how much.
pub struct Counted {
    /// The identity to accumulate under.
    pub key: CountKey,
    /// The amount to add: an op count for [`CountUnit::Ops`], a byte count for
    /// [`CountUnit::Bytes`].
    pub amount: u32,
}

/// How an operation participates in profiling.
///
/// The counting layer only ever sees this trait — never the concrete operation enums or opcodes.
/// The current IR implements it on [`Operation`] and its sub-enums; a future op representation
/// (e.g. pliron) implements it per op type without touching anything downstream.
///
/// [`count`](Trackable::count) is the single place that inspects the operation's result (`out`),
/// because in this IR the element type and vectorization live on the result value, not on the op
/// node. Structural queries like [`is_terminal`](Trackable::is_terminal) need no result and stay
/// separate, so callers can test them without a result in hand.
pub trait Trackable {
    /// This operation's counting contribution given its result value, or `None` if it is not
    /// counted. `out` is `None` for operations that produce no value.
    fn count(&self, out: Option<&Value>) -> Option<Counted>;

    /// Whether this operation ends a block and should trigger a flush of the accumulated counts.
    fn is_terminal(&self) -> bool {
        false
    }
}

impl Trackable for Operation {
    fn count(&self, out: Option<&Value>) -> Option<Counted> {
        match self {
            Operation::Arithmetic(op) => op.count(out),
            Operation::Memory(op) => op.count(out),
            _ => None,
        }
    }

    fn is_terminal(&self) -> bool {
        match self {
            Operation::Branch(op) => op.is_terminal(),
            _ => false,
        }
    }
}

impl Trackable for Arithmetic {
    fn count(&self, out: Option<&Value>) -> Option<Counted> {
        // Arithmetic is billed against its result's type and vectorization.
        let out = out?;
        Some(Counted {
            key: CountKey {
                name: Intern::from(self.op_code().name()),
                elem: out.elem_type(),
                unit: CountUnit::Ops,
            },
            amount: out.vector_size() as u32,
        })
    }
}

impl Trackable for Memory {
    fn count(&self, out: Option<&Value>) -> Option<Counted> {
        // The value whose element type/width we bill, plus the address space(s) the op touches.
        let (value, spaces): (&Value, [Option<AddressSpace>; 2]) = match self {
            Memory::Index(_) => {
                let out = out?;
                (out, [Some(out.address_space()), None])
            }
            Memory::Load(variable) => (variable, [Some(variable.address_space()), None]),
            Memory::Store(op) => (&op.value, [Some(op.ptr.address_space()), None]),
            Memory::CopyMemory(op) => (
                &op.source,
                [
                    Some(op.source.address_space()),
                    Some(op.target.address_space()),
                ],
            ),
        };

        // Local and shared traffic is free; only other address spaces count as moved bytes.
        let billable = spaces
            .iter()
            .flatten()
            .any(|space| !matches!(space, AddressSpace::Local | AddressSpace::Shared));
        if !billable {
            return None;
        }

        let elem = value.elem_type();
        Some(Counted {
            key: CountKey {
                name: Intern::from(self.op_code().name()),
                elem,
                unit: CountUnit::Bytes,
            },
            amount: (value.vector_size() * elem.size()) as u32,
        })
    }
}

impl Trackable for Branch {
    fn count(&self, _out: Option<&Value>) -> Option<Counted> {
        None
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Branch::Return)
    }
}

pub fn counter_stored_type() -> Type {
    Type::atomic(Type::scalar(ElemType::UInt(UIntKind::U32)))
}

fn display_u32(value: u32) -> String {
    format!("{value}")
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(|c| core::str::from_utf8(c).unwrap())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn fmt_counts(
    counts: &HashMap<CountKey, u32>,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let mut entries: Vec<(ElemType, CountUnit, &str, u32)> = counts
        .iter()
        .filter(|(_, c)| **c != 0)
        .map(|(k, &c)| (k.elem, k.unit, &*k.name, c))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(b.2)));

    if entries.is_empty() {
        return writeln!(f, "    (none)");
    }
    let mut cur_elem: Option<ElemType> = None;
    let mut cur_unit: Option<CountUnit> = None;
    for (elem, unit, name, count) in entries {
        if cur_elem != Some(elem) {
            writeln!(f, "{elem}")?;
            cur_elem = Some(elem);
            cur_unit = None;
        }
        if cur_unit != Some(unit) {
            writeln!(f, "└── {}", unit.suffix())?;
            cur_unit = Some(unit);
        }
        writeln!(
            f,
            "    └── {name}: {} {}",
            display_u32(count),
            unit.suffix()
        )?;
    }
    Ok(())
}
