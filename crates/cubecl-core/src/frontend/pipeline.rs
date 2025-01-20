//! This module exposes pipeling utilities for multi-stage asynchronous data copies
//! with latency hiding.
//! We call producers all threads that call producer_acquire and producer_commit,
//! and consumers threads that call consumer_wait and consumer_release.
//!
//! # Example
//! In this example, threads play both the role of producer and consumer
//!
//! ```rust, ignore
//! #[cube(launch)]
//! /// Calculate the sum of an array, using pipelining
//! fn pipelined_sum<F: Float>(
//!     input: &Array<Line<F>>,
//!     output: &mut Array<Line<F>>,
//!     #[comptime] batch_len: u32,
//! ) {
//!     let smem_size = 2 * batch_len;
//!     let num_batches = input.len() / batch_len;
//!     let mut shared_memory = SharedMemory::<F>::new_lined(smem_size, input.line_size());
//!     let pipeline = Pipeline::new();
//!
//!     let mut sum = Line::<F>::empty(input.line_size()).fill(F::new(0.));
//!
//!     // Copy the first batch to shared memory
//!     pipeline.producer_acquire();
//!     pipeline.memcpy_async(
//!         input.slice(0, batch_len),
//!         shared_memory.slice_mut(0, batch_len),
//!     );
//!     pipeline.producer_commit();
//!
//!     for input_batch in 1..num_batches {
//!         // Copy and compute index always alternate
//!         let copy_index = input_batch % 2;
//!         let compute_index = (input_batch + 1) % 2;
//!
//!         // Copy the next batch to shared memory
//!         pipeline.producer_acquire();
//!         pipeline.memcpy_async(
//!             input.slice(batch_len * input_batch, batch_len * (input_batch + 1)),
//!             shared_memory.slice_mut(batch_len * copy_index, batch_len * (copy_index + 1)),
//!         );
//!         pipeline.producer_commit();
//!
//!         // Compute the batch that is ready
//!         pipeline.consumer_wait();
//!         let compute_slice =
//!             shared_memory.slice(batch_len * compute_index, batch_len * (compute_index + 1));
//!         for i in 0..batch_len {
//!             sum += compute_slice[i];
//!         }
//!         pipeline.consumer_release();
//!     }
//!
//!     // Compute the last batch
//!     pipeline.consumer_wait();
//!     let compute_slice = shared_memory.slice(
//!         batch_len * ((num_batches + 1) % 2),
//!         batch_len * ((num_batches + 1) % 2 + 1),
//!     );
//!     for i in 0..batch_len {
//!         sum += compute_slice[i];
//!     }
//!     pipeline.consumer_release();
//!
//!     output[0] = sum;
//! }
//! ```

use std::marker::PhantomData;

use crate::{
    ir::{Instruction, Item, Operation, PipelineOps},
    unexpanded,
};

use super::{
    CubeContext, CubePrimitive, CubeType, ExpandElement, ExpandElementTyped, Init, IntoRuntime,
    Line, Slice, SliceMut,
};

/// A mechanism for managing a sequence of `memcpy_async`
/// For now, it only works at the Cube scope
#[derive(Clone, Copy)]
pub struct Pipeline<C: CubePrimitive> {
    _c: PhantomData<C>,
}

impl<C: CubePrimitive> IntoRuntime for Pipeline<C> {
    fn __expand_runtime_method(self, _context: &mut CubeContext) -> Self::ExpandType {
        panic!("Doesn't exist at runtime")
    }
}

impl<C: CubePrimitive> CubeType for Pipeline<C> {
    type ExpandType = PipelineExpand<C>;
}

impl<C: CubePrimitive> Init for PipelineExpand<C> {
    fn init(self, _context: &mut CubeContext) -> Self {
        self
    }
}

#[derive(Clone)]
/// Expand type of [Pipeline]
pub struct PipelineExpand<C: CubePrimitive> {
    elem: ExpandElement,
    _c: PhantomData<C>,
}

impl<C: CubePrimitive> Default for Pipeline<C> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl<C: CubePrimitive> Pipeline<C> {
    /// Create a pipeline instance
    pub fn new(_num_steps: u8) -> Self {
        Self { _c: PhantomData }
    }

    /// Copy the source slice to destination
    ///
    /// # Safety
    ///
    /// This will try to copy the whole source slice, so
    /// make sure source length <= destination length
    pub fn memcpy_async(&self, _source: Slice<Line<C>>, _destination: SliceMut<Line<C>>) {
        unexpanded!()
    }

    /// Reserves a specific stage for the producer to work on.
    pub fn producer_acquire(&self) {
        unexpanded!()
    }

    /// Signals that the producer is done and the stage is ready for the consumer.
    pub fn producer_commit(&self) {
        unexpanded!()
    }

    /// Waits until the producer has finished with the stage.
    pub fn consumer_wait(&self) {
        unexpanded!()
    }

    /// Frees the stage after the consumer is done using it.
    pub fn consumer_release(&self) {
        unexpanded!()
    }

    pub fn __expand_new(context: &mut CubeContext, num_steps: u8) -> PipelineExpand<C> {
        let elem = C::as_elem(context);
        let variable = context.create_pipeline(Item::new(elem), num_steps);
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

    pub fn __expand_consumer_wait(context: &mut CubeContext, expand: PipelineExpand<C>) {
        expand.__expand_consumer_wait_method(context);
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
    pub fn __expand_consumer_wait_method(&self, context: &mut CubeContext) {
        let pipeline = *self.elem;
        context.register(Instruction {
            out: None,
            operation: Operation::Pipeline(PipelineOps::ConsumerWait { pipeline }),
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
