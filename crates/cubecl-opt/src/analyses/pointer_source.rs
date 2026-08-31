use core::{any::type_name, cell::Ref, ops::Deref};

use ::pliron::{
    graph::walkers::{WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::op_cast,
    r#type::Typed,
};
use cubecl_environment::collections::{HashMap, HashSet};
use cubecl_ir::{
    AddressSpace,
    dialect::{base::OperationPtrExt, memory::DeclareVariableOp},
    interfaces::{MemoryEffect, MemoryEffects, TypedExt, aliasing::AliasingOp},
    prelude::*,
    types::PointerType,
};
use derive_new::new;
use pliron::{pass::Analysis, value::Value};

use crate::{BufferVisibility, MemoryResource};

#[derive(Default)]
pub struct Resources {
    pub memory_resources: HashSet<MemoryResource>,
}

impl Analysis for Resources {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn compute(op: Ptr<Operation>, ctx: &Context, _analyses: &mut AnalysisManager) -> Result<Self>
    where
        Self: Sized,
    {
        let mut this = Self::default();
        walk_op(
            ctx,
            &mut this,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, this, node| match node {
                IRNode::Operation(op) => {
                    let op_dyn = op.dyn_op(ctx);
                    if let Some(decl) = op_dyn.downcast_ref::<DeclareVariableOp>() {
                        this.analyze_declare(ctx, decl);
                    }
                }
                IRNode::BasicBlock(block) => {
                    for arg in block.deref(ctx).arguments() {
                        this.analyze_block_arg(ctx, arg);
                    }
                }
                _ => {}
            },
        );
        Ok(this)
    }
}

impl Resources {
    fn analyze_block_arg(&mut self, ctx: &Context, arg: Value) {
        let ty = arg.get_type(ctx).deref(ctx);
        if let Some(PointerType {
            inner,
            address_space,
        }) = ty.downcast_ref()
            && matches!(address_space, AddressSpace::Global(_))
        {
            let resource = MemoryResource {
                address_space: *address_space,
                value_ty: *inner,
                alignment: inner.align(ctx),
                root_ptr: arg,
            };
            self.memory_resources.insert(resource);
        }
    }

    fn analyze_declare(&mut self, ctx: &Context, declare: &DeclareVariableOp) {
        let root_ptr = declare.get_result(ctx);
        let resource = MemoryResource {
            address_space: declare.addr_space(ctx).0,
            value_ty: declare.value_ty(ctx).get_type(ctx),
            alignment: declare.alignment(ctx).0,
            root_ptr,
        };
        self.memory_resources.insert(resource);
    }
}

#[derive(Debug, new)]
pub struct PointerSource {
    /// The source memory of each pointer, propagated through copies
    pointer_sources: HashMap<Value, MemoryResource>,
}

impl Deref for PointerSource {
    type Target = HashMap<Value, MemoryResource>;

    fn deref(&self) -> &Self::Target {
        &self.pointer_sources
    }
}

impl Analysis for PointerSource {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn compute(op: Ptr<Operation>, ctx: &Context, analyses: &mut AnalysisManager) -> Result<Self>
    where
        Self: Sized,
    {
        let resources = analyses.get_analysis::<Resources>(op, ctx)?;
        let mem_resources = resources.memory_resources.iter();
        let mut this = Self::new(mem_resources.map(|it| (it.root_ptr, *it)).collect());
        walk_op(
            ctx,
            &mut this,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, this, node| {
                if let IRNode::Operation(op) = node {
                    let op_dyn = op.dyn_op(ctx);
                    if let Some(aliases) = op_cast::<dyn AliasingOp>(op_dyn.as_ref())
                        && let Some(source_ptr) = aliases.source_ptr(ctx)
                        && let Some(resource) = this.pointer_sources.get(&source_ptr)
                    {
                        this.pointer_sources
                            .insert(aliases.get_result(ctx), *resource);
                    }
                }
            },
        );
        Ok(this)
    }
}

#[derive(new, Debug)]
pub struct GlobalVisibility {
    pub visibility: HashMap<usize, BufferVisibility>,
}

struct GlobalVisibilityState<'a> {
    ptr_source: Ref<'a, PointerSource>,
    visibility: HashMap<usize, BufferVisibility>,
}

impl Analysis for GlobalVisibility {
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    fn compute(op: Ptr<Operation>, ctx: &Context, analyses: &mut AnalysisManager) -> Result<Self>
    where
        Self: Sized,
    {
        let visibility = {
            let resources = analyses.get_analysis::<Resources>(op, ctx)?;
            let mem_resources = resources.memory_resources.iter();
            let globals = mem_resources.filter_map(|res| match res.address_space {
                AddressSpace::Global(id) => Some((id, Default::default())),
                _ => None,
            });
            globals.collect()
        };

        let ptr_source = analyses.get_analysis::<PointerSource>(op, ctx)?;

        let mut state = GlobalVisibilityState {
            ptr_source,
            visibility,
        };
        walk_op(
            ctx,
            &mut state,
            &WALKCONFIG_PREORDER_FORWARD,
            op,
            |ctx, state, node| {
                if let IRNode::Operation(op) = node {
                    let op_dyn = op.dyn_op(ctx);
                    if let Some(effects) = op_cast::<dyn MemoryEffects>(op_dyn.as_ref()) {
                        for effect in effects.memory_effects(ctx) {
                            match effect {
                                MemoryEffect::Read(affects) => state.check_read(ctx, affects),
                                MemoryEffect::Write(affects) => state.check_write(ctx, affects),
                                // Inline asm: it names no pointer, so it can
                                // touch any buffer the kernel holds.
                                MemoryEffect::ReadAll => state.read_all(),
                                MemoryEffect::WriteAll => state.write_all(),
                            }
                        }
                    }
                }
            },
        );
        Ok(Self::new(state.visibility))
    }
}

/// What an effect through a value can touch. Because this analysis feeds
/// correctness downstream — a buffer stamped `Dead` gets no write tracking —
/// the failure direction matters: an access nobody can attribute must widen
/// visibility, never drop out of it.
enum Touches {
    /// One global buffer, by binding index.
    Global(usize),
    /// No global buffer: the value provably lives elsewhere — a shared or
    /// local pointer, or a register value like a matrix fragment.
    Nothing,
}

impl GlobalVisibilityState<'_> {
    /// Attribute the effect's value: by traced source when the chain is
    /// known, and by the value's own type otherwise. A pointer's type carries
    /// its address space — and for globals, the binding index — so an access
    /// through a pointer [`PointerSource`] could not follow is still pinned
    /// to the one buffer its type names instead of being dropped.
    fn touches(&self, ctx: &Context, value: Value) -> Touches {
        if let Some(resource) = self.ptr_source.get(&value) {
            return match resource.address_space {
                AddressSpace::Global(idx) => Touches::Global(idx),
                _ => Touches::Nothing,
            };
        }
        let ty = value.get_type(ctx);
        match ty.deref(ctx).downcast_ref::<PointerType>() {
            Some(PointerType {
                address_space: AddressSpace::Global(idx),
                ..
            }) => Touches::Global(*idx),
            _ => Touches::Nothing,
        }
    }

    fn check_read(&mut self, ctx: &Context, value: Value) {
        if let Touches::Global(idx) = self.touches(ctx, value)
            && let Some(visibility) = self.visibility.get_mut(&idx)
        {
            visibility.readable = true;
        }
    }

    fn check_write(&mut self, ctx: &Context, value: Value) {
        if let Touches::Global(idx) = self.touches(ctx, value)
            && let Some(visibility) = self.visibility.get_mut(&idx)
        {
            visibility.writable = true;
        }
    }

    fn read_all(&mut self) {
        for visibility in self.visibility.values_mut() {
            visibility.readable = true;
        }
    }

    fn write_all(&mut self) {
        for visibility in self.visibility.values_mut() {
            visibility.writable = true;
        }
    }
}
