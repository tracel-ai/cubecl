use alloc::vec::Vec;
use cubecl_ir::{
    AddressSpace, Allocator, Arithmetic, AtomicBinaryOperands, AtomicOp, BinaryOperands,
    ConstantValue, CountKey, ElemType, IndexOperands, Instruction, Memory, Operation, StorageType,
    StoreOperands, Trackable, Type, UIntKind, Value,
};
use hashbrown::{HashMap, hash_map::Entry};

type BufferId = Value;

const LOCAL_COUNTER_TYPE: Type = Type::Scalar(StorageType::Scalar(ElemType::UInt(UIntKind::U32)));

fn constant_value(amount: usize) -> Value {
    Value::constant(ConstantValue::UInt(amount as u64), LOCAL_COUNTER_TYPE)
}

#[derive(Clone, Debug, Default, derive_new::new)]
pub struct CompilerProfiler {
    global_counter: Option<BufferId>,
    locals: HashMap<CountKey, (BufferId, usize)>,
    current_index: usize,
}

impl CompilerProfiler {
    pub fn set_counter(&mut self, counter: BufferId) {
        self.global_counter = Some(counter)
    }

    pub fn profile_operation(
        &mut self,
        operation: &impl Trackable,
        out: Option<&Value>,
        allocator: &Allocator,
    ) -> Vec<Instruction> {
        // Terminals carry no result, so this must be checked before touching `out`.
        if operation.is_terminal() {
            let mut instructions = Vec::new();
            for (local_counter, index) in self.locals.values() {
                instructions.extend(self.flush_instructions(local_counter, index, allocator));
            }
            return instructions;
        }

        let Some(counted) = operation.count(out) else {
            return Vec::new();
        };

        let local_counter = self.local_counter(counted.key, allocator);
        Vec::from(self.increment_local_instructions(
            &local_counter,
            counted.amount as usize,
            allocator,
        ))
    }

    pub fn profile(&self, allocator: &Allocator) -> (Vec<Instruction>, Vec<Instruction>) {
        let mut declares = Vec::new();
        let mut flushes = Vec::new();

        self.locals.values().for_each(|(local_counter, index)| {
            declares.extend(self.declare_instructions(local_counter));
            flushes.extend(self.flush_instructions(local_counter, index, allocator));
        });

        (declares, flushes)
    }

    fn increment_local_instructions(
        &self,
        local_counter: &BufferId,
        amount: usize,
        allocator: &Allocator,
    ) -> [Instruction; 3] {
        let loaded = allocator.create_value(LOCAL_COUNTER_TYPE);
        let added = allocator.create_value(LOCAL_COUNTER_TYPE);

        [
            Instruction::new(Memory::Load(*local_counter), loaded),
            Instruction::new(
                Arithmetic::Add(BinaryOperands {
                    lhs: loaded,
                    rhs: constant_value(amount),
                }),
                added,
            ),
            Instruction::no_out(Memory::Store(StoreOperands {
                ptr: *local_counter,
                value: added,
            })),
        ]
    }

    fn declare_instructions(&self, local_counter: &BufferId) -> [Instruction; 2] {
        [
            Instruction::new(
                Operation::DeclareVariable {
                    value_ty: LOCAL_COUNTER_TYPE,
                    addr_space: AddressSpace::Local,
                    alignment: LOCAL_COUNTER_TYPE.align(),
                },
                *local_counter,
            ),
            Instruction::no_out(Memory::Store(StoreOperands {
                ptr: *local_counter,
                value: constant_value(0),
            })),
        ]
    }

    fn flush_instructions(
        &self,
        local_counter: &BufferId,
        index: &usize,
        allocator: &Allocator,
    ) -> [Instruction; 3] {
        let global = self
            .global_counter
            .as_ref()
            .expect("Global counter should be available");

        let loaded = allocator.create_value(LOCAL_COUNTER_TYPE);
        let slot_ptr =
            allocator.create_value(Type::pointer(global.value_type(), global.address_space()));
        let discard = allocator.create_value(LOCAL_COUNTER_TYPE);

        [
            Instruction::new(Memory::Load(*local_counter), loaded),
            Instruction::new(
                Memory::Index(IndexOperands {
                    list: *global,
                    index: constant_value(*index),
                    checked: false,
                    unroll_factor: 1,
                }),
                slot_ptr,
            ),
            Instruction::new(
                AtomicOp::Add(AtomicBinaryOperands {
                    ptr: slot_ptr,
                    value: loaded,
                }),
                discard,
            ),
        ]
    }

    fn local_counter(&mut self, key: CountKey, allocator: &Allocator) -> BufferId {
        match self.locals.entry(key) {
            Entry::Occupied(entry) => entry.get().0,
            Entry::Vacant(entry) => {
                let index = self.current_index;
                self.current_index += 1;
                let counter =
                    allocator.create_value(Type::pointer(LOCAL_COUNTER_TYPE, AddressSpace::Local));
                entry.insert((counter, index));
                counter
            }
        }
    }

    /// The slot → key decoding table, indexed by dense slot. Its length is the number of used
    /// counters, i.e. the size the global profiling buffer must be. Empty when nothing was
    /// tracked.
    pub fn profile_map(&self) -> Vec<CountKey> {
        let mut entries: Vec<(usize, CountKey)> = self
            .locals
            .iter()
            .map(|(key, (_, slot))| (*slot, *key))
            .collect();
        entries.sort_by_key(|(slot, _)| *slot);
        entries.into_iter().map(|(_, key)| key).collect()
    }
}
