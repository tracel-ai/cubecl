use core::{any::type_name, cell::Ref, ops::Deref};

use ::pliron::{
    graph::walkers::{WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    op::op_cast,
    r#type::Typed,
};
use cubecl_ir::{
    AddressSpace,
    dialect::{base::OperationPtrExt, memory::DeclareVariableOp},
    interfaces::{MemoryEffect, MemoryEffects, TypedExt, aliasing::AliasingOp},
    prelude::*,
    types::PointerType,
};
use derive_new::new;
use hashbrown::{HashMap, HashSet};
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
                                MemoryEffect::Read(affects) => state.check_read(affects),
                                MemoryEffect::Write(affects) => state.check_write(affects),
                                MemoryEffect::ReadAll | MemoryEffect::WriteAll => {
                                    // Technically need to handle it, but let's leave it for now
                                }
                            }
                        }
                    }
                }
            },
        );
        Ok(Self::new(state.visibility))
    }
}

impl GlobalVisibilityState<'_> {
    fn check_read(&mut self, ptr: Value) {
        if let Some(resource) = self.ptr_source.get(&ptr)
            && let AddressSpace::Global(idx) = resource.address_space
            && let Some(visibility) = self.visibility.get_mut(&idx)
        {
            visibility.readable = true;
        }
    }

    fn check_write(&mut self, ptr: Value) {
        if let Some(resource) = self.ptr_source.get(&ptr)
            && let AddressSpace::Global(idx) = resource.address_space
            && let Some(visibility) = self.visibility.get_mut(&idx)
        {
            visibility.writable = true;
        }
    }
}
