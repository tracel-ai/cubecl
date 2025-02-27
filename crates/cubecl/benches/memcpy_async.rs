use cubecl::prelude::barrier::{Barrier, BarrierLevel};
use cubecl::prelude::*;
use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimestampsResult, TimingMethod};
use cubecl::frontend::Float;
use cubecl::future;
use cubecl_linalg::tensor::TensorHandle;

#[cube]
trait CopyStrategy: Send + Sync + 'static {
    type Barrier: CubeType + Copy + Clone;

    fn barrier() -> Self::Barrier;

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    );

    fn wait(_barrier: Self::Barrier);
}

#[derive(CubeType)]
struct DummyCopy {}

#[cube]
impl CopyStrategy for DummyCopy {
    type Barrier = ();

    fn barrier() -> Self::Barrier {
        ()
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        _barrier: Self::Barrier,
        #[comptime] _config: Config,
    ) {
        for i in 0..source.len() {
            destination[i] = source[i];
        }
    }

    fn wait(_barrier: Self::Barrier) {
        // do nothing
    }
}

#[derive(CubeType)]
struct CoalescedCopy {}
#[cube]
impl CopyStrategy for CoalescedCopy {
    type Barrier = ();

    fn barrier() -> Self::Barrier {
        ()
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        _barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        let num_units = config.num_planes * config.plane_dim;
        let num_copies_per_unit = source.len() / num_units;
        for i in 0..num_copies_per_unit {
            let pos = UNIT_POS + i * num_units;
            destination[pos] = source[pos];
        }
    }

    fn wait(_barrier: Self::Barrier) {
        // do nothing
    }
}

#[derive(CubeType)]
struct MemcpyAsyncDuplicated {}
#[cube]
impl<E: Float> CopyStrategy<E> for MemcpyAsyncDuplicated {
    type Barrier = Barrier<E>;

    fn barrier() -> Self::Barrier {
        Barrier::<E>::new(BarrierLevel::cube(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        barrier.memcpy_async(source, destination)
    }

    fn wait(barrier: Self::Barrier) {
        barrier.wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncDispatched {}
#[cube]
impl<E: Float> CopyStrategy<E> for MemcpyAsyncDispatched {
    type Barrier = Barrier<E>;

    fn barrier() -> Self::Barrier {
        Barrier::<E>::new(BarrierLevel::cube(0u32))
    }

    fn memcpy(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        barrier.memcpy_async(source, destination)
    }

    fn wait(barrier: Self::Barrier) {
        barrier.wait();
    }
}

#[derive(CubeType)]
struct MemcpyAsyncCooperative {}
#[cube]
impl<E: Float> CopyStrategy<E> for MemcpyAsyncCooperative {
    type Barrier = Barrier<E>;

    fn barrier() -> Self::Barrier {
        Barrier::<E>::new(BarrierLevel::cube(0u32))
    }

    fn memcpy(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier,
        #[comptime] config: Config,
    ) {
        barrier.memcpy_async(source, destination)
    }

    fn wait(barrier: Self::Barrier) {
        barrier.wait();
    }
}

#[cube]
trait ComputeTask: Send + Sync + 'static {
    fn compute<E: Float>(data: &Slice<Line<E>>);
}

#[derive(CubeType)]
struct DummyCompute {}
#[cube]
impl ComputeTask for DummyCompute {
    fn compute<E: Float>(_data: &Slice<Line<E>>) {
        // do nothing
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Config {
    plane_dim: u32,
    num_planes: u32,
}

#[cube(launch_unchecked)]
fn memcpy_test<E: Float, Cpy: CopyStrategy<E>, Cpt: ComputeTask>(
    input: &Tensor<Line<E>>,
    #[comptime] config: Config,
) {
    let data_count = input.shape(0);
    let smem_size = 1024;

    let mut smem = SharedMemory::<E>::new_lined(smem_size, 1u32);
    let num_iterations = (data_count + smem_size - 1) / smem_size;

    let barrier = Cpy::barrier();

    for i in 0..num_iterations {
        let start = i * smem_size;
        let end = start + smem_size;

        Cpy::memcpy(
            &input.slice(start, end),
            &mut smem.to_slice_mut(),
            barrier,
            config,
        );

        Cpt::compute(&smem.to_slice());

        Cpy::wait(barrier);
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum CopyStrategyEnum {
    Dummy,
    CoalescedCopy,
    MemcpyAsyncDuplicated,
    MemcpyAsyncDispatched,
    MemcpyAsyncCooperative,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum ComputeTaskEnum {
    Dummy,
}

fn launch_ref<R: Runtime, E: Float>(
    strategy: CopyStrategyEnum,
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
) {
    let cube_count = CubeCount::Static(1, 1, 1);
    let plane_dim = 32;
    let num_planes = 8;
    let cube_dim = CubeDim::new_2d(plane_dim, num_planes);
    let config = Config {
        plane_dim,
        num_planes,
    };

    unsafe {
        match strategy {
            CopyStrategyEnum::Dummy => {
                memcpy_test::launch_unchecked::<E, DummyCopy, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::CoalescedCopy => {
                memcpy_test::launch_unchecked::<E, CoalescedCopy, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncDuplicated => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncDuplicated, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncDispatched => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncDispatched, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncCooperative => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncCooperative, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    config,
                )
            }
        }
    }
}

impl<R: Runtime, E: Float> Benchmark for MemcpyAsyncBench<R, E> {
    type Args = TensorHandle<R, E>;

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);

        TensorHandle::zeros(&client, vec![self.data_count])
    }

    fn execute(&self, input: Self::Args) {
        launch_ref::<R, E>(self.strategy, &self.client, &input.as_ref());
    }

    fn name(&self) -> String {
        format!(
            "memcpy_async-{}-{}-{:?}",
            R::name(),
            E::as_elem_native_unchecked(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn sync_elapsed(&self) -> TimestampsResult {
        future::block_on(self.client.sync_elapsed())
    }
}

#[allow(dead_code)]
struct MemcpyAsyncBench<R: Runtime, E> {
    data_count: usize,
    strategy: CopyStrategyEnum,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
fn run<R: Runtime, E: Float>(device: R::Device, strategy: CopyStrategyEnum) {
    let client = R::client(&device);

    // for data_count in [10000, 100000, 1000000, 10000000] {
    for data_count in [10000000] {
        let bench = MemcpyAsyncBench::<R, E> {
            data_count,
            strategy: strategy.clone(),
            client: client.clone(),
            device: device.clone(),
            _e: PhantomData,
        };
        println!("Data count: {data_count:?}, strategy: {strategy:?}");
        println!("{}", bench.name());
        println!("{}", bench.run(TimingMethod::Full));
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        use half::f16;

        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), CopyStrategyEnum::Dummy);
        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), CopyStrategyEnum::CoalescedCopy);
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncDuplicated,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncDispatched,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncCooperative,
        );
    }
}
