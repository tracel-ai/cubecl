use core::marker::PhantomData;

use variadics_please::all_tuples;

pub trait KeyGenerator<K, Inputs>: 'static {
    fn generate(&self, inputs: &Inputs) -> K;
}

pub trait IntoKeyGenerator<K, Inputs, Marker> {
    type Generator: KeyGenerator<K, Inputs>;

    fn into_key_gen(self) -> Self::Generator;
}

pub struct FunctionKeyGenerator<F, Marker> {
    func: F,
    _marker: PhantomData<Marker>,
}

impl<K, Inputs, Marker: 'static, F> KeyGenerator<K, Inputs> for FunctionKeyGenerator<F, Marker>
where
    F: FunctionKeygen<K, Inputs, Marker>,
{
    fn generate(&self, inputs: &Inputs) -> K {
        self.func.execute(inputs)
    }
}

pub trait FunctionKeygen<K, Inputs, Marker>: 'static {
    fn execute(&self, inputs: &Inputs) -> K;
}

impl<K, Inputs, Marker: 'static, F> IntoKeyGenerator<K, Inputs, Marker> for F
where
    F: FunctionKeygen<K, Inputs, Marker>,
{
    type Generator = FunctionKeyGenerator<F, Marker>;

    fn into_key_gen(self) -> Self::Generator {
        FunctionKeyGenerator {
            func: self,
            _marker: PhantomData,
        }
    }
}

macro_rules! impl_keygen {
    ($($param:ident),*) => {
        #[allow(unused_parens)]
        impl<K: 'static, Func, $($param: Clone + Send + 'static,)*> FunctionKeygen<K, ($($param),*), fn($(&$param),*) -> K> for Func
            where Func: Send + Sync + 'static,
            for<'a> &'a Func: Fn($(&$param),*) -> K
        {
            #[allow(non_snake_case, clippy::too_many_arguments)]
            #[inline]
            fn execute(&self, ($($param),*): &($($param),*)) -> K {
                fn call_inner<Out, $($param,)*>(
                    f: impl Fn($(&$param,)*) -> Out,
                    $($param: &$param,)*
                ) -> Out {
                    f($($param,)*)
                }
                call_inner(self, $($param),*)
            }
        }
    };
}

all_tuples!(impl_keygen, 0, 12, I);
