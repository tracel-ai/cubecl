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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OpPath(Intern<[Intern<str>]>);

impl OpPath {
    pub fn new(segments: &[&str]) -> Self {
        let segments: Vec<Intern<str>> = segments.iter().copied().map(Intern::from).collect();
        Self(Intern::from(segments.as_slice()))
    }
    /// The segments, outermost first.
    pub fn segments(&self) -> &[Intern<str>] {
        &self.0
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
    pub name: OpPath,
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
                name: OpPath::new(&[family::<Self>(), self.op_code().name()]),
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
                name: OpPath::new(&[family::<Self>(), self.op_code().name()]),
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

fn family<T>() -> &'static str {
    let name = core::any::type_name::<T>();
    name.rsplit("::").next().unwrap_or(name)
}

pub fn fmt_counts(
    counts: &HashMap<CountKey, u32>,
    f: &mut core::fmt::Formatter<'_>,
) -> core::fmt::Result {
    let mut entries: Vec<(ElemType, CountUnit, &[Intern<str>], u32)> = counts
        .iter()
        .filter(|(_, c)| **c != 0)
        .map(|(k, &c)| (k.elem, k.unit, k.name.segments(), c))
        .collect();

    // Sort by Elem, then strictly by Path (names), then Unit
    // Grouping by path ensures our tree algorithm correctly renders branches.
    entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(b.2)).then(a.1.cmp(&b.1)));

    if entries.is_empty() {
        return writeln!(f, "    (none)");
    }

    // 1. Calculate max lengths for vertical column alignment
    let mut max_op_len = 0;
    let mut max_count_len = 0;
    let mut formatted_counts = Vec::with_capacity(entries.len());

    for entry in &entries {
        let names = entry.2;
        let count = entry.3;

        // The operation name is the leaf node (last item in the path)
        let op_len = names.last().map(|s| s.len()).unwrap_or(6); // 6 is "(root)".len()
        if op_len > max_op_len {
            max_op_len = op_len;
        }

        let count_str = display_u32(count);
        if count_str.len() > max_count_len {
            max_count_len = count_str.len();
        }
        formatted_counts.push(count_str);
    }

    let mut cur_elem: Option<ElemType> = None;
    let mut is_last_at_depth: Vec<bool> = Vec::new(); // Tracks indentation levels

    // 2. Render the tree
    for (entry_index, entry) in entries.iter().enumerate() {
        let elem = &entry.0;
        let unit = &entry.1;
        let names = entry.2;

        // Print root elements without indentation
        if cur_elem.as_ref() != Some(elem) {
            writeln!(f, "{elem}")?;
            cur_elem = Some(*elem);
            is_last_at_depth.clear();
        }

        let mut common_len = 0;
        if entry_index > 0 && entries[entry_index - 1].0 == *elem {
            let prev_names = entries[entry_index - 1].2;
            for (p, n) in prev_names.iter().zip(names.iter()) {
                if p == n {
                    common_len += 1;
                } else {
                    break;
                }
            }
        }

        // If the path is an exact duplicate of the previous entry (e.g. same path, different unit)
        // Step back one depth level to ensure the leaf node prints again.
        if common_len == names.len() && !names.is_empty() {
            common_len -= 1;
        }

        let count_str = &formatted_counts[entry_index];

        if names.is_empty() {
            writeln!(
                f,
                "└── {:<width$} {:>count_width$} {}",
                "(root):",
                count_str,
                unit.suffix(),
                width = max_op_len + 1,
                count_width = max_count_len
            )?;
            continue;
        }

        for depth in common_len..names.len() {
            // Print background indentation structure mapped from parent relationships
            for p in 0..depth {
                if *is_last_at_depth.get(p).unwrap_or(&true) {
                    write!(f, "    ")?;
                } else {
                    write!(f, "│   ")?;
                }
            }

            let parent_path = &names[0..depth];
            let mut is_last = true;

            // Look ahead to evaluate if the current node has remaining siblings
            for next_entry in &entries[entry_index + 1..] {
                if next_entry.0 != *elem {
                    break;
                }
                let next_names = next_entry.2;

                if next_names.starts_with(parent_path) {
                    if next_names.len() > depth && next_names[depth] != names[depth] {
                        is_last = false; // Found a sibling
                        break;
                    }
                    if depth == names.len() - 1 && next_names == names {
                        is_last = false; // Identical leaf node (usually from duplicate/multiple units)
                        break;
                    }
                } else {
                    break; // Out of the shared parent branch
                }
            }

            // Record this depth's 'last' status for its children to use
            if depth >= is_last_at_depth.len() {
                is_last_at_depth.push(is_last);
            } else {
                is_last_at_depth[depth] = is_last;
            }
            is_last_at_depth.truncate(depth + 1);

            let prefix = if is_last { "└── " } else { "├── " };

            if depth == names.len() - 1 {
                // Leaf Node: Append the colon and format with column alignment
                let op_with_colon = format!("{}:", names[depth]);
                writeln!(
                    f,
                    "{}{:<width$} {:>count_width$} {}",
                    prefix,
                    op_with_colon,
                    count_str,
                    unit.suffix(),
                    width = max_op_len + 1,
                    count_width = max_count_len
                )?;
            } else {
                // Intermediate Parent Node
                writeln!(f, "{}{}", prefix, names[depth])?;
            }
        }
    }

    Ok(())
}
