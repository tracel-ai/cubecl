use crate::{associative_scan, instructions::ScanInstruction};
use cubecl::prelude::*;
use cubecl_core as cubecl;
use cubecl_std::tensor::TensorHandle;
use rand::{Rng, SeedableRng, distr::Uniform};

#[macro_export]
macro_rules! testgen_scan_simple {
    () => {
        mod scan_simple {
            use super::*;
            use cubecl_scan::*;

            $crate::testgen_scan_simple!(@group: [ty=[i8, u8, i16, u16, i32, u32], sizes=[1, 10, 100, 128, 256, 1_000, 4097, 1_000_000]]: {
                Add: |a, b| a.wrapping_add(b);
            });
            $crate::testgen_scan_simple!(@group: [ty=[f32], sizes=[1, 10, 100, 128, 256, 1_000, 4097]]: {
                Add: |a, b| a + b;
            });
        }
    };
    (@group: [ty=[$($ty:ty),*], sizes=$sizes:expr]: $rest:tt) => {
        $(
            $crate::testgen_scan_simple!(@group: [ty=$ty, sizes=$sizes, false]: $rest);
            $crate::testgen_scan_simple!(@group: [ty=$ty, sizes=$sizes, true]: $rest);
        )*
    };
    (@group: [ty=$ty:ty, sizes=$sizes:expr, $inclusive:literal]: {
        $( $instr:ty : $op:expr ; )*
    }) => {
        $(
            paste::paste! {
                #[test]
                fn [<test_ $ty:lower _ $inclusive:lower _ $instr:lower>]() {
                    for size in $sizes {
                        let client = TestRuntime::client(&Default::default());
                        let test = cubecl_scan::tests::simple::TestCase {
                            shape: vec![size],
                            stride: vec![1],
                            axis: 0,
                            inclusive: $inclusive,
                        };
                        test.test_scan::<TestRuntime, $ty, instructions::$instr>(&client, $op);
                    }
                }
            }
        )*
    };
}

#[derive(Debug)]
pub struct TestCase {
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub axis: usize,
    pub inclusive: bool,
}

impl TestCase {
    pub fn test_scan<R: Runtime, N: CubePrimitive + CubeElement + Numeric, I: ScanInstruction>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        op: impl Fn(N, N) -> N,
    ) {
        let len = self.shape.iter().product::<usize>();
        let data = rand::rngs::StdRng::seed_from_u64(1234)
            .sample_iter(Uniform::<i64>::new(1, 20).unwrap())
            .take(len)
            .map(|v| N::from_int(v))
            .collect::<Vec<_>>();
        let expected = self.reference_scan(&data, N::from_int(0), op);

        let handle = client.create(N::as_bytes(&data));
        let input = TensorHandle::<R, N>::new(handle, vec![len], self.stride.clone());
        let output = TensorHandle::<R, N>::empty(client, vec![len]);

        let res = associative_scan::<R, N, I>(
            &client,
            input.as_ref(),
            output.as_ref(),
            0,
            self.inclusive,
        );

        if res.is_err() {
            return;
        }

        let output_data = client.read_one(output.handle);
        let output_data = &N::from_bytes(&output_data)[..len];

        assert_eq!(&expected[..], output_data);
    }

    fn reference_scan<T: Copy>(&self, data: &[T], start: T, op: impl Fn(T, T) -> T) -> Vec<T> {
        data.iter()
            .scan(start, |acc, v| {
                let mut res = *acc;
                *acc = op(*acc, *v);
                if self.inclusive {
                    res = *acc;
                }

                Some(res)
            })
            .collect()
    }
}
