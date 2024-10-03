use std::collections::HashMap;

use petgraph::{graph::NodeIndex, visit::EdgeRef, Direction};
use rspirv::spirv::Word;
use serde::{Deserialize, Serialize};

use crate::{SpirvCompiler, SpirvTarget};

use super::{BasicBlock, Optimizer};

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct ExpandState {
    label: Option<Word>,
    bindings: HashMap<(u16, u8), Word>,
}

pub struct PhiEntry {
    block: Word,
    value: Word,
}
pub struct PhiInstruction {
    id: (u16, u8),
    entries: Vec<PhiEntry>,
}

impl BasicBlock {
    pub fn label<T: SpirvTarget>(&mut self, b: &mut SpirvCompiler<T>) -> Word {
        *self.expand.label.get_or_insert_with(|| b.id())
    }

    pub fn write_binding<T: SpirvTarget>(&mut self, b: &mut SpirvCompiler<T>, id: (u16, u8)) {
        self.expand.bindings.insert(id, b.id());
    }

    pub fn read_binding(&mut self, id: (u16, u8)) -> Word {
        self.expand.bindings[&id]
    }
}

impl Optimizer {
    pub fn merge_vars<T: SpirvTarget>(
        &mut self,
        b: &mut SpirvCompiler<T>,
        block: NodeIndex,
    ) -> Vec<PhiInstruction> {
        let incoming_edges: Vec<_> = self
            .program
            .edges_directed(block, Direction::Incoming)
            .map(|edge| edge.source())
            .collect();

        if incoming_edges.len() == 1 {
            let edge = incoming_edges[0];
            self.program[block].expand.bindings = self.program[edge].expand.bindings.clone();
            return vec![];
        }

        let mut merged = HashMap::new();
        for predecessor in incoming_edges.iter().copied() {
            merged.extend(self.program[predecessor].expand.bindings.clone());
        }

        let phi_instructions = self.program[block]
            .phi_nodes
            .clone()
            .into_iter()
            .map(|id| {
                let entries = incoming_edges
                    .iter()
                    .map(|predecessor| {
                        let binding = self.program[*predecessor].expand.bindings[&id];
                        let label = self.program[*predecessor].label(b);
                        PhiEntry {
                            block: label,
                            value: binding,
                        }
                    })
                    .collect();
                merged.insert(id, b.id());
                PhiInstruction { id, entries }
            })
            .collect();

        self.program[block].expand.bindings = merged;

        phi_instructions
    }
}
