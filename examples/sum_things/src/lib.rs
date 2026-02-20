use cubecl::{features::Plane, prelude::*, server::Handle};
use std::marker::PhantomData;

#[cube(launch_unchecked)]
fn sum_basic<F: Float>(input: &Array<F>, output: &mut Array<F>, #[comptime] end: Option<usize>) {
    let unroll = end.is_some();
    let end = end.unwrap_or_else(|| input.len());

    let mut sum = F::new(0.0);

    #[unroll(unroll)]
    for i in 0..end {
        sum += input[i];
    }

    output[UNIT_POS as usize] = sum;
}

#[cube(launch_unchecked)]
fn sum_subgroup<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] subgroup: bool,
    #[comptime] end: Option<usize>,
) {
    if subgroup {
        output[UNIT_POS as usize] = plane_sum(input[UNIT_POS as usize]);
    } else {
        sum_basic(input, output, end);
    }
}

#[cube]
trait SumKind: 'static + Send + Sync {
    fn sum<F: Float>(input: &Slice<F>, #[comptime] end: Option<usize>) -> F;
}

struct SumBasic;
struct SumPlane;

#[cube]
impl SumKind for SumBasic {
    fn sum<F: Float>(input: &Slice<F>, #[comptime] end: Option<usize>) -> F {
        let unroll = end.is_some();
        let end = end.unwrap_or_else(|| input.len());

        let mut sum = F::new(0.0);

        #[unroll(unroll)]
        for i in 0..end {
            sum += input[i];
        }

        sum
    }
}

#[cube]
impl SumKind for SumPlane {
    fn sum<F: Float>(input: &Slice<F>, #[comptime] _end: Option<usize>) -> F {
        plane_sum(input[UNIT_POS as usize])
    }
}

#[cube(launch_unchecked)]
fn sum_trait<F: Float, K: SumKind>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] end: Option<usize>,
) {
    output[UNIT_POS as usize] = K::sum(&input.to_slice(), end);
}

#[cube]
trait CreateSeries: 'static + Send + Sync {
    type SumKind: SumKind;

    fn execute<F: Float>(input: &Slice<F>, #[comptime] end: Option<usize>) -> F;
}

#[cube(launch_unchecked)]
fn series<F: Float, S: CreateSeries>(
    input: &Array<F>,
    output: &mut Array<F>,
    #[comptime] end: Option<usize>,
) {
    output[UNIT_POS as usize] = S::execute(&input.to_slice(), end);
}

struct SumThenMul<K: SumKind> {
    _p: PhantomData<K>,
}

#[cube]
impl<K: SumKind> CreateSeries for SumThenMul<K> {
    type SumKind = K;

    fn execute<F: Float>(input: &Slice<F>, #[comptime] end: Option<usize>) -> F {
        let val = Self::SumKind::sum(input, end);
        val * input[UNIT_POS as usize]
    }
}

fn launch_basic<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    len: usize,
) {
    unsafe {
        sum_basic::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts::<f32>(input, len, 1),
            ArrayArg::from_raw_parts::<f32>(output, len, 1),
            Some(len),
        )
    }
}

fn launch_subgroup<R: Runtime>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    len: usize,
) {
    unsafe {
        sum_subgroup::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts::<f32>(input, len, 1),
            ArrayArg::from_raw_parts::<f32>(output, len, 1),
            client.properties().features.plane.contains(Plane::Ops),
            Some(len),
        )
    }
}

fn launch_trait<R: Runtime, K: SumKind>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    len: usize,
) {
    unsafe {
        sum_trait::launch_unchecked::<f32, K, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts::<f32>(input, len, 1),
            ArrayArg::from_raw_parts::<f32>(output, len, 1),
            Some(len),
        )
    }
}

fn launch_series<R: Runtime, S: CreateSeries>(
    client: &ComputeClient<R>,
    input: &Handle,
    output: &Handle,
    len: usize,
) {
    unsafe {
        series::launch_unchecked::<f32, S, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(len as u32),
            ArrayArg::from_raw_parts::<f32>(input, len, 1),
            ArrayArg::from_raw_parts::<f32>(output, len, 1),
            Some(len),
        )
    }
}

#[derive(Debug)]
enum KernelKind {
    Basic,
    Plane,
    TraitSum,
    SeriesSumThenMul,
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 10., 1., 5.];
    let len = input.len();

    let output = client.empty(input.len() * core::mem::size_of::<f32>());
    let input = client.create_from_slice(f32::as_bytes(input));

    for kind in [
        KernelKind::Basic,
        KernelKind::Plane,
        KernelKind::TraitSum,
        KernelKind::SeriesSumThenMul,
    ] {
        match kind {
            KernelKind::Basic => launch_basic(&client, &input, &output, len),
            KernelKind::Plane => launch_subgroup(&client, &input, &output, len),
            KernelKind::TraitSum => {
                // When using trait, it's normally a good idea to check if the variation can be
                // executed.
                if client.properties().features.plane.contains(Plane::Ops) {
                    launch_trait::<R, SumPlane>(&client, &input, &output, len)
                } else {
                    launch_trait::<R, SumBasic>(&client, &input, &output, len)
                }
            }
            KernelKind::SeriesSumThenMul => {
                if client.properties().features.plane.contains(Plane::Ops) {
                    launch_series::<R, SumThenMul<SumPlane>>(&client, &input, &output, len)
                } else {
                    launch_series::<R, SumThenMul<SumBasic>>(&client, &input, &output, len)
                }
            }
        }
        let bytes = client.read_one(output.clone()).unwrap();
        let output = f32::from_bytes(&bytes);

        println!("[{:?} - {kind:?}]\n {output:?}", R::name(&client));
    }
}
