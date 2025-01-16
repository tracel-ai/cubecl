//! This module exposes pipeling utilities for multi-stage asynchronous data copies
//! with latency hiding.
//!
//! # Example
//!
//! ```rust, ignore
//! #[cube(launch)]
//! pub fn example(lhs: &Array<F16>, rhs: &Array<F16>, out: &mut Array<F32>) {
//!     let a = cmma::Matrix::<F16>::new(
//!         cmma::MatrixIdent::A,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::RowMajor,
//!     );
//!     let b = cmma::Matrix::<F16>::new(
//!         cmma::MatrixIdent::B,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::ColMajor,
//!     );
//!     let c = cmma::Matrix::<F32>::new(
//!         cmma::MatrixIdent::Accumulator,
//!         16,
//!         16,
//!         16,
//!         cmma::MatrixLayout::Undefined,
//!     );
//!     cmma::fill::<F32>(&c, F32::new(0.0));
//!     cmma::load::<F16>(&a, lhs.as_slice(), u32::new(16));
//!     cmma::load::<F16>(&b, rhs.as_slice(), u32::new(16));
//!
//!     cmma::execute::<F16, F16, F32, F32>(&a, &b, &c, &c);
//!
//!     cmma::store::<F32>(
//!         out.as_slice_mut(),
//!         &c,
//!         u32::new(16),
//!         cmma::MatrixLayout::RowMajor,
//!     );
//! }
//! ```

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

impl<C: CubePrimitive> Default for Pipeline<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: CubePrimitive> Pipeline<C> {
    pub fn new() -> Self {
        Self { _c: PhantomData }
    }

    pub fn memcpy_async(&self, _source: Slice<Line<C>>, _destination: SliceMut<Line<C>>) {
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
        let pipeline = *self.elem;
        context.register(Instruction {
            out: None,
            operation: Operation::Pipeline(PipelineOps::ProducerAcquire { pipeline }),
        });
    }
    pub fn __expand_producer_commit_method(&self, context: &mut CubeContext) {
        let pipeline = *self.elem;
        context.register(Instruction {
            out: None,
            operation: Operation::Pipeline(PipelineOps::ProducerCommit { pipeline }),
        });
    }
    pub fn __expand_consumer_await_method(&self, context: &mut CubeContext) {
        let pipeline = *self.elem;
        context.register(Instruction {
            out: None,
            operation: Operation::Pipeline(PipelineOps::ConsumerAwait { pipeline }),
        });
    }
    pub fn __expand_consumer_release_method(&self, context: &mut CubeContext) {
        let pipeline = *self.elem;
        context.register(Instruction {
            out: None,
            operation: Operation::Pipeline(PipelineOps::ConsumerRelease { pipeline }),
        });
    }
}
