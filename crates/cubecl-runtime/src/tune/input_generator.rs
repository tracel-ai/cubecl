use core::marker::PhantomData;

use variadics_please::all_tuples;

pub trait InputGenerator<K, Inputs>: 'static {
    fn generate(&self, key: &K, inputs: &Inputs) -> Inputs;
}

pub trait IntoInputGenerator<K, Inputs, Marker> {
    type Generator: InputGenerator<K, Inputs>;

    fn into_input_gen(self) -> Self::Generator;
}

pub struct FunctionInputGenerator<F, Marker> {
    func: F,
    _marker: PhantomData<Marker>,
}

impl<K, Inputs, Marker: 'static, F> InputGenerator<K, Inputs> for FunctionInputGenerator<F, Marker>
where
    F: FunctionInputGen<K, Inputs, Marker>,
{
    fn generate(&self, key: &K, inputs: &Inputs) -> Inputs {
        self.func.execute(key, inputs)
    }
}

pub trait FunctionInputGen<K, Inputs, Marker>: 'static {
    fn execute(&self, key: &K, inputs: &Inputs) -> Inputs;
}

impl<K, Inputs, Marker: 'static, F> IntoInputGenerator<K, Inputs, Marker> for F
where
    F: FunctionInputGen<K, Inputs, Marker>,
{
    type Generator = FunctionInputGenerator<F, Marker>;

    fn into_input_gen(self) -> Self::Generator {
        FunctionInputGenerator {
            func: self,
            _marker: PhantomData,
        }
    }
}

macro_rules! impl_input_gen {
    ($($param:ident),*) => {
        #[allow(unused_parens, clippy::unused_unit)]
        impl<K: 'static, Func, $($param: Clone + Send + 'static,)*> FunctionInputGen<K, ($($param),*), fn(&K, $(&$param),*) -> ($($param),*)> for Func
            where Func: Send + Sync + 'static,
            for<'a> &'a Func: Fn(&K, $(&$param),*) -> ($($param),*)
        {
            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[inline]
            fn execute(&self, key: &K, ($($param),*): &($($param),*)) -> ($($param),*) {
                fn call_inner<K, $($param,)*>(
                    f: impl Fn(&K, $(&$param,)*) -> ($($param),*),
                    key: &K,
                    $($param: &$param,)*
                ) -> ($($param),*) {
                    f(key, $($param,)*)
                }
                call_inner(self, key, $($param),*)
            }
        }
    };
}

all_tuples!(impl_input_gen, 0, 16, I);
