use std::marker::PhantomData;

use cubecl::prelude::*;
use cubecl_core::{self as cubecl, unexpanded};
use cubecl_std::tensor::r#virtual::{
    ReadWrite, VirtualTensor, VirtualTensorOperations, VirtualTensorOperationsExpand,
};

pub trait ReduceDType {
    type In: Numeric;
    type Out: Numeric;
}

impl<In: Numeric, Out: Numeric> ReduceDType for (In, Out) {
    type In = In;
    type Out = Out;
}

#[cube]
#[allow(dead_code)]
pub trait ReduceArgs: Send + Sync + 'static + Clone {
    type Input<E: Numeric>: LaunchArg + CubeType;
    type Output<E: Numeric>: LaunchArg + CubeType;
    type State<P: ReduceDType>: CubeType;

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In>,
        output: &mut Self::Output<P::Out>,
    ) -> Self::State<P>;

    fn read_input<P: ReduceDType>(state: &Self::State<P>, index: u32) -> Line<P::In>;
    fn read_output<P: ReduceDType>(state: &Self::State<P>, index: u32) -> Line<P::Out>;

    fn write_output<P: ReduceDType>(state: &mut Self::State<P>, index: u32, value: Line<P::Out>);

    fn len_input<P: ReduceDType>(state: &Self::State<P>) -> u32;
    fn len_output<P: ReduceDType>(state: &Self::State<P>) -> u32;

    fn buffer_len_input<P: ReduceDType>(state: &Self::State<P>) -> u32;
    fn buffer_len_output<P: ReduceDType>(state: &Self::State<P>) -> u32;

    fn rank_input<P: ReduceDType>(state: &Self::State<P>) -> u32;
    fn rank_output<P: ReduceDType>(state: &Self::State<P>) -> u32;

    fn shape_input<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32;
    fn shape_output<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32;

    fn stride_input<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32;
    fn stride_output<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32;
}

#[cube]
pub fn init_tensors<RA: ReduceArgs, In: Numeric, Out: Numeric>(
    input: &RA::Input<In>,
    output: &mut RA::Output<Out>,
) -> (VirtualTensor<In>, VirtualTensor<Out, ReadWrite>) {
    let mut state = RA::init_state::<(In, Out)>(input, output);

    let input = TensorArg::new_input(&state);
    let mut output = TensorArg::new_output(&mut state);

    let input = VirtualTensor::<In>::new::<TensorArg<(In, Out), RA, Input>>(&input);
    let output =
        VirtualTensor::<Out, ReadWrite>::new::<TensorArg<(In, Out), RA, Output>>(&mut output);

    (input, output)
}

#[derive(Clone)]
pub struct TensorArgs;

#[cube]
impl ReduceArgs for TensorArgs {
    type Input<EG: Numeric> = Tensor<Line<EG>>;
    type Output<EG: Numeric> = Tensor<Line<EG>>;
    type State<P: ReduceDType> = (*const Tensor<Line<P::In>>, *mut Tensor<Line<P::Out>>);

    fn init_state<P: ReduceDType>(
        input: &Self::Input<P::In>,
        output: &mut Self::Output<P::Out>,
    ) -> Self::State<P> {
        (input, output)
    }

    fn read_input<P: ReduceDType>(state: &Self::State<P>, index: u32) -> Line<P::In> {
        unsafe { (*state.0)[index] }
    }

    fn read_output<P: ReduceDType>(state: &Self::State<P>, index: u32) -> Line<P::Out> {
        unsafe { (*state.1)[index] }
    }

    fn write_output<P: ReduceDType>(state: &mut Self::State<P>, index: u32, value: Line<P::Out>) {
        unsafe { (*state.1)[index] = value }
    }

    fn buffer_len_input<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        unsafe { (*state.0).buffer_len() }
    }

    fn buffer_len_output<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        unsafe { (*state.1).buffer_len() }
    }

    fn len_input<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        unsafe { (*state.0).len() }
    }

    fn len_output<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        unsafe { (*state.1).len() }
    }
    fn rank_input<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        unsafe { (*state.0).rank() }
    }

    fn rank_output<P: ReduceDType>(state: &Self::State<P>) -> u32 {
        unsafe { (*state.1).rank() }
    }

    fn shape_input<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        unsafe { (*state.0).shape(dim) }
    }

    fn shape_output<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        unsafe { (*state.1).shape(dim) }
    }

    fn stride_input<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        unsafe { (*state.0).stride(dim) }
    }

    fn stride_output<P: ReduceDType>(state: &Self::State<P>, dim: u32) -> u32 {
        unsafe { (*state.1).stride(dim) }
    }
}

pub struct Input;
pub struct Output;

pub struct TensorArg<P: ReduceDType, RA: ReduceArgs, Tag> {
    _state: *mut RA::State<P>,
    tag: PhantomData<Tag>,
}

pub struct TensorArgExpand<P: ReduceDType, RA: ReduceArgs, Tag> {
    state: <RA::State<P> as CubeType>::ExpandType,
    tag: PhantomData<Tag>,
}

impl<P: ReduceDType, RA: ReduceArgs> TensorArg<P, RA, Input> {
    pub fn new_input(_state: &RA::State<P>) -> Self {
        unexpanded!()
    }
    pub fn __expand_new_input(
        _scope: &mut Scope,
        state: <RA::State<P> as CubeType>::ExpandType,
    ) -> TensorArgExpand<P, RA, Input> {
        TensorArgExpand {
            state,
            tag: PhantomData,
        }
    }
}

impl<P: ReduceDType, RA: ReduceArgs> TensorArg<P, RA, Output> {
    pub fn new_output(_state: &mut RA::State<P>) -> Self {
        unexpanded!()
    }
    pub fn __expand_new_output(
        _scope: &mut Scope,
        state: <RA::State<P> as CubeType>::ExpandType,
    ) -> TensorArgExpand<P, RA, Output> {
        TensorArgExpand {
            state,
            tag: PhantomData,
        }
    }
}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperations<P::Out> for TensorArg<P, RA, Output> {}
impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperations<P::In> for TensorArg<P, RA, Input> {}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperationsExpand<P::In>
    for TensorArgExpand<P, RA, Input>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<P::In>> {
        RA::__expand_read_input(scope, self.state.clone(), index)
    }

    fn __expand_write_method(
        &self,
        _scope: &mut Scope,
        _index: ExpandElementTyped<u32>,
        _value: ExpandElementTyped<Line<P::In>>,
    ) {
        unreachable!("Can't write to input")
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        RA::__expand_shape_input(scope, self.state.clone(), axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        RA::__expand_stride_input(scope, self.state.clone(), axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        RA::__expand_rank_input(scope, self.state.clone())
    }
    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        RA::__expand_len_input(scope, self.state.clone())
    }
    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        RA::__expand_buffer_len_input(scope, self.state.clone())
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<u32>,
        _end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<P::In>>> {
        panic!("Unsupported")
    }
}

impl<P: ReduceDType, RA: ReduceArgs> VirtualTensorOperationsExpand<P::Out>
    for TensorArgExpand<P, RA, Output>
{
    fn __expand_read_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Line<P::Out>> {
        RA::__expand_read_output(scope, self.state.clone(), index)
    }

    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        index: ExpandElementTyped<u32>,
        value: ExpandElementTyped<Line<P::Out>>,
    ) {
        RA::__expand_write_output(scope, self.state.clone(), index, value)
    }

    fn __expand_shape_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        RA::__expand_shape_output(scope, self.state.clone(), axis)
    }

    fn __expand_stride_method(
        &self,
        scope: &mut Scope,
        axis: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        RA::__expand_stride_output(scope, self.state.clone(), axis)
    }

    fn __expand_rank_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        RA::__expand_rank_output(scope, self.state.clone())
    }

    fn __expand_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        RA::__expand_len_output(scope, self.state.clone())
    }
    fn __expand_buffer_len_method(&self, scope: &mut Scope) -> ExpandElementTyped<u32> {
        RA::__expand_buffer_len_output(scope, self.state.clone())
    }

    fn __expand_read_window_method(
        &self,
        _context: &mut Scope,
        _start: ExpandElementTyped<u32>,
        _end: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<Slice<Line<P::Out>>> {
        panic!("Unsupported")
    }
}

mod __tensor_arg {
    use super::*;

    impl<P: ReduceDType, RA: ReduceArgs, Tag> CubeType for TensorArg<P, RA, Tag> {
        type ExpandType = TensorArgExpand<P, RA, Tag>;
    }

    impl<P: ReduceDType, RA: ReduceArgs, Tag> Init for TensorArgExpand<P, RA, Tag> {
        fn init(self, _scope: &mut Scope) -> Self {
            self
        }
    }

    impl<P: ReduceDType, RA: ReduceArgs, Tag> CubeDebug for TensorArgExpand<P, RA, Tag> {}
    impl<P: ReduceDType, RA: ReduceArgs, Tag> Clone for TensorArgExpand<P, RA, Tag> {
        fn clone(&self) -> Self {
            Self {
                state: self.state.clone(),
                tag: self.tag,
            }
        }
    }
}
