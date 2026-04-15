use alloc::string::String;
use alloc::vec::Vec;
use variadics_please::all_tuples;

/// An input to an autotuned operation.
///
/// Implementors distinguish between real-execution cloning ([`Clone::clone`])
/// and benchmark-execution cloning ([`AutotuneInput::fork`]).
///
/// On targets where autotune benchmarks run synchronously (native), `fork` can
/// match `clone` exactly. On targets where benchmarks run in the background
/// (wasm), `fork` should return a transient handle that keeps the underlying
/// buffer alive without incrementing any reference count that an external
/// consumer (e.g. a fusion context) watches to know when the real inputs are
/// no longer in use. The default implementation delegates to `clone`, which is
/// correct for any type that has no such external watcher.
pub trait AutotuneInput: Clone + Send + 'static {
    /// Fork this input for benchmark execution during autotuning.
    fn fork(&self) -> Self {
        self.clone()
    }
}

macro_rules! impl_autotune_input_primitive {
    ($($ty:ty),* $(,)?) => {
        $(impl AutotuneInput for $ty {})*
    }
}

impl_autotune_input_primitive!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char, String,
);

impl<T: AutotuneInput> AutotuneInput for Vec<T> {
    fn fork(&self) -> Self {
        self.iter().map(AutotuneInput::fork).collect()
    }
}

impl<T: AutotuneInput> AutotuneInput for Option<T> {
    fn fork(&self) -> Self {
        self.as_ref().map(AutotuneInput::fork)
    }
}

macro_rules! impl_autotune_input_tuple {
    ($($param:ident),*) => {
        #[allow(non_snake_case, clippy::unused_unit)]
        impl<$($param: AutotuneInput),*> AutotuneInput for ($($param,)*) {
            fn fork(&self) -> Self {
                let ($($param,)*) = self;
                ($($param.fork(),)*)
            }
        }
    }
}

all_tuples!(impl_autotune_input_tuple, 0, 13, I);
