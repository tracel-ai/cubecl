use cubecl::prelude::barrier::{Barrier, BarrierLevel};
use cubecl::prelude::*;
use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimestampsResult, TimingMethod};
use cubecl::frontend::Float;
use cubecl::future;
use cubecl_linalg::tensor::TensorHandle;

#[cube]
trait CopyStrategy: Send + Sync + 'static {
    fn memcpy<E: Float>(
        source: &Slice<E>,
        destination: &mut SliceMut<E>,
        barrier: Barrier<E>,
        #[comptime] config: Config,
    );
}

#[derive(CubeType)]
struct DummyCopy {}
#[cube]
impl CopyStrategy for DummyCopy {
    fn memcpy<E: Float>(
        source: &Slice<E>,
        destination: &mut SliceMut<E>,
        barrier: Barrier<E>,
        #[comptime] _config: Config,
    ) {
        for i in 0..source.len() {
            destination[i] = source[i];
        }
    }
}

#[derive(CubeType)]
struct CoalescedCopy {}
#[cube]
impl CopyStrategy for CoalescedCopy {
    fn memcpy<E: Float>(
        source: &Slice<E>,
        destination: &mut SliceMut<E>,
        barrier: Barrier<E>,
        #[comptime] config: Config,
    ) {
        let num_units = config.num_planes * config.plane_dim;
        let num_copies_per_unit = source.len() / num_units;
        for i in 0..num_copies_per_unit {
            let pos = UNIT_POS + i * num_units;
            destination[pos] = source[pos];
        }
    }
}

#[derive(CubeType)]
struct MemcpyAsyncDuplicated {}
#[cube]
impl CopyStrategy for MemcpyAsyncDuplicated {
    fn memcpy<E: Float>(
        source: &Slice<E>,
        destination: &mut SliceMut<E>,
        barrier: Barrier<E>,
        #[comptime] config: Config,
    ) {
    }
}

#[derive(CubeType)]
struct MemcpyAsyncDispatched {}
#[cube]
impl CopyStrategy for MemcpyAsyncDispatched {
    fn memcpy<E: Float>(
        source: &Slice<E>,
        destination: &mut SliceMut<E>,
        barrier: Barrier<E>,
        #[comptime] config: Config,
    ) {
    }
}

#[derive(CubeType)]
struct MemcpyAsyncCooperative {}
#[cube]
impl CopyStrategy for MemcpyAsyncCooperative {
    fn memcpy<E: Float>(
        source: &Slice<E>,
        destination: &mut SliceMut<E>,
        barrier: Barrier<E>,
        #[comptime] config: Config,
    ) {
    }
}

#[cube]
trait ComputeTask: Send + Sync + 'static {
    fn compute<E: Float>(data: &Slice<E>);
}

#[derive(CubeType)]
struct DummyCompute {}
#[cube]
impl ComputeTask for DummyCompute {
    fn compute<E: Float>(_data: &Slice<E>) {
        // do nothing
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Config {
    plane_dim: u32,
    num_planes: u32,
}

#[cube(launch_unchecked)]
fn memcpy_test<E: Float, Cpy: CopyStrategy, Cpt: ComputeTask>(
    input: &Tensor<E>,
    #[comptime] config: Config,
) {
    let data_count = input.shape(0);
    let smem_size = 1024;

    let mut smem = SharedMemory::<E>::new(smem_size);
    let num_iterations = (data_count + smem_size - 1) / smem_size;

    let barrier = Barrier::<E>::new(BarrierLevel::cube(0u32));

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

        barrier.wait();
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum CopyStrategyEnum {
    Dummy,
    CoalescedCopy,
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

    for data_count in [10000, 100000, 1000000, 10000000] {
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

        run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), CopyStrategyEnum::Dummy);
        run::<cubecl::wgpu::WgpuRuntime, f32>(Default::default(), CopyStrategyEnum::CoalescedCopy);
        run::<cubecl::wgpu::WgpuRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncDuplicated,
        );
        run::<cubecl::wgpu::WgpuRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncDispatched,
        );
        run::<cubecl::wgpu::WgpuRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncCooperative,
        );
    }
}
