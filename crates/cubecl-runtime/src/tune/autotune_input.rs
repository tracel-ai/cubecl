/// An input to an autotuned operation.
///
/// Distinguishes between cloning for benchmark execution during tuning and
/// cloning for the real execution path. On targets where benchmarks run
/// synchronously, both can match. On targets where benchmarks run in the
/// background (wasm), `clone_for_benchmark` should return a transient handle
/// that does not increment any reference count watched by an external consumer
/// (e.g. fusion).
pub trait AutotuneInput: Send + 'static {
    /// Clone this input for benchmark execution during autotuning.
    fn clone_for_benchmark(&self) -> Self;
    /// Clone this input for the real execution path.
    fn clone_for_execution(&self) -> Self;
}

impl<T: Clone + Send + 'static> AutotuneInput for T {
    fn clone_for_execution(&self) -> Self {
        self.clone()
    }
    fn clone_for_benchmark(&self) -> Self {
        self.clone()
    }
}

//   /// A set of inputs to an autotuned operation, implemented for tuples of [`AutotuneInput`].
//   pub trait AutotuneInputs: Send + 'static {
//       /// Clone this input for benchmark execution during autotuning.
//       fn clone_for_benchmark(&self) -> Self;
//       /// Clone this input for the real execution path.
//       fn clone_for_execution(&self) -> Self;
//   }
//
//   macro_rules! impl_autotune_inputs_tuple {
//       ($($name:ident),+) => {
//           impl<$($name: AutotuneInput),+> AutotuneInputs for ($($name,)+) {
//               fn clone_for_benchmark(&self) -> Self {
//                   #[allow(non_snake_case)]
//                   let ($($name,)+) = self;
//                   ($($name.clone_for_benchmark(),)+)
//               }
//               fn clone_for_execution(&self) -> Self {
//                   #[allow(non_snake_case)]
//                   let ($($name,)+) = self;
//                   ($($name.clone_for_execution(),)+)
//               }
//           }
//       };
//   }
//
//   impl AutotuneInputs for () {
//       fn clone_for_benchmark(&self) -> Self {}
//       fn clone_for_execution(&self) -> Self {}
//   }
//
//   impl_autotune_inputs_tuple!(T0);
//   impl_autotune_inputs_tuple!(T0, T1);
//   impl_autotune_inputs_tuple!(T0, T1, T2);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4, T5);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4, T5, T6);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4, T5, T6, T7);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
//   impl_autotune_inputs_tuple!(T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);
//
//   impl<T: AutotuneInput> AutotuneInputs for alloc::vec::Vec<T> {
//       fn clone_for_benchmark(&self) -> Self {
//           todo!()
//       }
//
//       fn clone_for_execution(&self) -> Self {
//           todo!()
//       }
//   }
