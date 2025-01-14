use std::marker::PhantomData;

use crate::{
    ir::{Instruction, Item, Operation, PipelineOps},
    unexpanded,
};

use super::{CubeContext, CubePrimitive, ExpandElement, ExpandElementTyped, Line, Slice, SliceMut};

pub struct Pipeline<C: CubePrimitive> {
    _c: PhantomData<C>,
}

#[derive(Clone)]
pub struct PipelineExpand<C: CubePrimitive> {
    elem: ExpandElement,
    _c: PhantomData<C>,
}

impl<C: CubePrimitive> Pipeline<C> {
    pub fn new() -> Self {
        Self { _c: PhantomData }
    }

    pub fn memcpy_async(&self, source: Slice<Line<C>>, destination: SliceMut<Line<C>>) {
        unexpanded!()
    }

    pub fn producer_acquire(&self) {
        unexpanded!()
    }
    pub fn producer_commit(&self) {
        unexpanded!()
    }
    pub fn consumer_await(&self) {
        unexpanded!()
    }
    pub fn consumer_release(&self) {
        unexpanded!()
    }

    pub fn __expand_new(context: &mut CubeContext) -> PipelineExpand<C> {
        let elem = C::as_elem(context);
        let variable = context.create_pipeline(Item::new(elem));
        PipelineExpand {
            elem: variable,
            _c: PhantomData,
        }
    }

    pub fn __expand_memcpy_async(
        context: &mut CubeContext,
        expand: PipelineExpand<C>,
        source: ExpandElementTyped<Slice<Line<C>>>,
        destination: ExpandElementTyped<SliceMut<Line<C>>>,
    ) {
        expand.__expand_memcpy_async_method(context, source, destination);
    }

    pub fn __expand_producer_acquire(context: &mut CubeContext, expand: PipelineExpand<C>) {
        expand.__expand_producer_acquire_method(context);
    }

    pub fn __expand_producer_commit(context: &mut CubeContext, expand: PipelineExpand<C>) {
        expand.__expand_producer_commit_method(context);
    }

    pub fn __expand_consumer_await(context: &mut CubeContext, expand: PipelineExpand<C>) {
        expand.__expand_consumer_await_method(context);
    }

    pub fn __expand_consumer_release(context: &mut CubeContext, expand: PipelineExpand<C>) {
        expand.__expand_consumer_release_method(context);
    }
}

impl<C: CubePrimitive> PipelineExpand<C> {
    pub fn __expand_memcpy_async_method(
        &self,
        context: &mut CubeContext,
        source: ExpandElementTyped<Slice<Line<C>>>,
        destination: ExpandElementTyped<SliceMut<Line<C>>>,
    ) {
        let pipeline = *self.elem;
        let source = *source.expand;
        let destination = *destination.expand;

        let mem_copy = PipelineOps::MemCopyAsync {
            pipeline,
            source,
            destination,
        };

        context.register(Instruction {
            out: None,
            operation: Operation::Pipeline(mem_copy),
        });
    }

    pub fn __expand_producer_acquire_method(&self, context: &mut CubeContext) {
        context.register(Instruction {
            out: None,
            operation: Operation::Pipeline(PipelineOps::ProducerAcquire(pipeline)),
        });
    }
    pub fn __expand_producer_commit_method(&self, context: &mut CubeContext) {
        todo!()
    }
    pub fn __expand_consumer_await_method(&self, context: &mut CubeContext) {
        todo!()
    }
    pub fn __expand_consumer_release_method(&self, context: &mut CubeContext) {
        todo!()
    }
}
