#![allow(unknown_lints)] // `manual_div_ceil` only appeared in 1.83
#![allow(clippy::manual_div_ceil)]

use cubecl::prelude::barrier::{Barrier, BarrierLevel};
use cubecl::prelude::*;
use std::marker::PhantomData;

use cubecl::benchmark::{Benchmark, TimingMethod};
use cubecl::frontend::Float;
use cubecl::future;
use cubecl_linalg::tensor::TensorHandle;

#[cube]
trait ComputeTask: Send + Sync + 'static {
    fn compute<E: Float>(
        input: &Slice<Line<E>>,
        acc: &mut Array<Line<E>>,
        #[comptime] config: Config,
    );

    fn to_output<E: Float>(
        acc: &mut Array<Line<E>>,
        output: &mut SliceMut<Line<E>>,
        #[comptime] config: Config,
    );
}

#[derive(CubeType)]
struct DummyCompute {}
#[cube]
impl ComputeTask for DummyCompute {
    fn compute<E: Float>(
        input: &Slice<Line<E>>,
        acc: &mut Array<Line<E>>,
        #[comptime] config: Config,
    ) {
        // An offset to make sure units need the data loaded by other units
        let offset = 256;

        let position = (UNIT_POS * config.acc_len + offset) % config.smem_size;
        for i in 0..config.acc_len {
            acc[i] += input[position + i];
        }
    }

    fn to_output<E: Float>(
        acc: &mut Array<Line<E>>,
        output: &mut SliceMut<Line<E>>,
        #[comptime] config: Config,
    ) {
        let position = UNIT_POS * config.acc_len;
        for i in 0..config.acc_len {
            acc[i] += output[position + i];
        }
    }
}

#[cube]
trait CopyStrategy: Send + Sync + 'static {
    type Barrier<E: Float>: CubeType + Copy + Clone;

    fn barrier<E: Float>() -> Self::Barrier<E>;

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    );

    fn wait<E: Float>(_barrier: Self::Barrier<E>);
}

#[derive(CubeType)]
/// All units perform all loads, duplicatively
/// Not async
struct DummyCopy {}
#[cube]
impl CopyStrategy for DummyCopy {
    type Barrier<E: Float> = ();

    fn barrier<E: Float>() -> Self::Barrier<E> {}

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        _barrier: Self::Barrier<E>,
        #[comptime] _config: Config,
    ) {
        for i in 0..source.len() {
            destination[i] = source[i];
        }
    }

    fn wait<E: Float>(_barrier: Self::Barrier<E>) {
        sync_cube();
    }
}

#[derive(CubeType)]
/// Units worked in a coalesced manner to share the loading
/// Not async
struct CoalescedCopy {}
#[cube]
impl CopyStrategy for CoalescedCopy {
    type Barrier<E: Float> = ();

    fn barrier<E: Float>() -> Self::Barrier<E> {}

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        _barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    ) {
        let num_units = config.num_planes * config.plane_dim;
        let num_copies_per_unit = source.len() / num_units;
        for i in 0..num_copies_per_unit {
            let pos = UNIT_POS + i * num_units;
            destination[pos] = source[pos];
        }
    }

    fn wait<E: Float>(_barrier: Self::Barrier<E>) {
        sync_cube();
    }
}

#[derive(CubeType)]
/// All units will call one slice that encompasses all data
/// using a per-unit memcpy_async
///
/// Probably very slow as all units perform the copy duplicatively
struct MemcpyAsyncSingleSliceDuplicatedAll {}
#[cube]
impl CopyStrategy for MemcpyAsyncSingleSliceDuplicatedAll {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] _config: Config,
    ) {
        barrier.memcpy_async(source, destination)
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// One unit is elected to call one slice that encompasses all data
/// using a per-unit memcpy_async
///
/// Probably very slow as one unit does everything
struct MemcpyAsyncSingleSliceElected {}
#[cube]
impl CopyStrategy for MemcpyAsyncSingleSliceElected {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] _config: Config,
    ) {
        if UNIT_POS == 0 {
            barrier.memcpy_async(source, destination)
        }
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// One unit is elected to call one slice that encompasses all data
/// using a cooperative memcpy_async
///
/// Probably the best for contiguous data
struct MemcpyAsyncSingleSliceElectedCooperative {}
#[cube]
impl CopyStrategy for MemcpyAsyncSingleSliceElectedCooperative {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_coop(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] _config: Config,
    ) {
        if UNIT_POS == 0 {
            barrier.memcpy_async(source, destination)
        }
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// The slice is divided by the number of planes, then each subslice
/// is handled by a plane.
/// In this case all units of the plane call memcpy
struct MemcpyAsyncSplitPlaneDuplicatedUnit {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitPlaneDuplicatedUnit {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() / config.num_planes;
        let start = UNIT_POS_Y * sub_length;
        let end = start + sub_length;

        barrier.memcpy_async(
            &source.slice(start, end),
            &mut destination.slice_mut(start, end),
        )
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// The slice is divided by the number of planes, then each subslice
/// is handled by a plane.
/// In this case one unit of the plane is elected to call memcpy
struct MemcpyAsyncSplitPlaneElectedUnit {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitPlaneElectedUnit {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() / config.num_planes;
        let start = UNIT_POS_Y * sub_length;
        let end = start + sub_length;

        if UNIT_POS_X == 0 {
            barrier.memcpy_async(
                &source.slice(start, end),
                &mut destination.slice_mut(start, end),
            )
        }
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// The slice is divided by the number of planes (arbitrartily), but each
/// unit calls each slice
///
/// Probably the worst
struct MemcpyAsyncSplitDuplicatedAll {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitDuplicatedAll {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() / config.num_planes;
        for i in 0..config.num_planes {
            let start = i * sub_length;
            let end = start + sub_length;

            barrier.memcpy_async(
                &source.slice(start, end),
                &mut destination.slice_mut(start, end),
            )
        }
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// The slice is divided by the number of planes n (arbitrarily)
/// Then the first n units in absolute each perform a call
struct MemcpyAsyncSplitLargeUnitWithIdle {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitLargeUnitWithIdle {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() / config.num_planes;

        if UNIT_POS < config.num_planes {
            let start = UNIT_POS * sub_length;
            let end = start + sub_length;

            barrier.memcpy_async(
                &source.slice(start, end),
                &mut destination.slice_mut(start, end),
            )
        }
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// The slice is divided into single elements.
/// then all units make coalesced calls for a single element,
/// looping cyclicly until the end
struct MemcpyAsyncSplitSmallUnitCoalescedLoop {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitSmallUnitCoalescedLoop {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    ) {
        let num_units = config.num_planes * config.plane_dim;
        let num_loops = source.len() / num_units;

        for i in 0..num_loops {
            let start = UNIT_POS + i * num_units;
            let end = start + 1;

            barrier.memcpy_async(
                &source.slice(start, end),
                &mut destination.slice_mut(start, end),
            )
        }
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(CubeType)]
/// The slice is divided by the number of units,
/// then all units make one coalesced call of the length of the subslices
///
/// Probably the fastest for a non contiguous series of contiguous slices
struct MemcpyAsyncSplitMediumUnitCoalescedOnce {}
#[cube]
impl CopyStrategy for MemcpyAsyncSplitMediumUnitCoalescedOnce {
    type Barrier<E: Float> = Barrier<E>;

    fn barrier<E: Float>() -> Self::Barrier<E> {
        Barrier::<E>::new(BarrierLevel::cube_manual(0u32))
    }

    fn memcpy<E: Float>(
        source: &Slice<Line<E>>,
        destination: &mut SliceMut<Line<E>>,
        barrier: Self::Barrier<E>,
        #[comptime] config: Config,
    ) {
        let sub_length = source.len() / (config.num_planes * config.plane_dim);

        let start = UNIT_POS * sub_length;
        let end = start + sub_length;

        barrier.memcpy_async(
            &source.slice(start, end),
            &mut destination.slice_mut(start, end),
        )
    }

    fn wait<E: Float>(barrier: Self::Barrier<E>) {
        barrier.arrive_and_wait();
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct Config {
    plane_dim: u32,
    num_planes: u32,
    smem_size: u32,
    acc_len: u32,
    double_buffering: bool,
}

#[cube(launch_unchecked)]
fn memcpy_test<E: Float, Cpy: CopyStrategy, Cpt: ComputeTask>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    #[comptime] config: Config,
) {
    if config.double_buffering {
        memcpy_test_single_buffer::<E, Cpy, Cpt>(input, output, config);
    } else {
        memcpy_test_double_buffer::<E, Cpy, Cpt>(input, output, config);
    }
}

#[cube]
fn memcpy_test_single_buffer<E: Float, Cpy: CopyStrategy, Cpt: ComputeTask>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    #[comptime] config: Config,
) {
    let data_count = input.shape(0);
    let mut acc = Array::<Line<E>>::new(config.acc_len);
    let num_iterations = (data_count + config.smem_size - 1) / config.smem_size;

    let mut smem = SharedMemory::<E>::new_lined(config.smem_size, 1u32);
    let barrier = Cpy::barrier();

    for i in 0..num_iterations {
        let start = i * config.smem_size;
        let end = start + config.smem_size;

        Cpy::memcpy(
            &input.slice(start, end),
            &mut smem.to_slice_mut(),
            barrier,
            config,
        );

        Cpy::wait(barrier);

        Cpt::compute(&smem.to_slice(), &mut acc, config);
    }

    Cpy::wait(barrier);
    Cpt::compute(&smem.to_slice(), &mut acc, config);
    Cpt::to_output(&mut acc, &mut output.to_slice_mut(), config);
}

#[cube]
fn memcpy_test_double_buffer<E: Float, Cpy: CopyStrategy, Cpt: ComputeTask>(
    input: &Tensor<Line<E>>,
    output: &mut Tensor<Line<E>>,
    #[comptime] config: Config,
) {
    let data_count = input.shape(0);
    let mut smem1 = SharedMemory::<E>::new_lined(config.smem_size, 1u32);
    let mut smem2 = SharedMemory::<E>::new_lined(config.smem_size, 1u32);
    let mut acc = Array::<Line<E>>::new(config.acc_len);
    let num_iterations = (data_count + config.smem_size - 1) / config.smem_size;

    let barrier1 = Cpy::barrier();
    let barrier2 = Cpy::barrier();

    for i in 0..num_iterations {
        let start = i * config.smem_size;
        let end = if start + config.smem_size < data_count {
            start + config.smem_size
        } else {
            data_count
        };

        if i % 2 == 0 {
            Cpy::memcpy(
                &input.slice(start, end),
                &mut smem1.to_slice_mut(),
                barrier1,
                config,
            );
            if i > 0 {
                Cpy::wait(barrier2);
                Cpt::compute(&smem2.to_slice(), &mut acc, config);
            }
        } else {
            Cpy::memcpy(
                &input.slice(start, end),
                &mut smem2.to_slice_mut(),
                barrier2,
                config,
            );

            Cpy::wait(barrier1);
            Cpt::compute(&smem1.to_slice(), &mut acc, config);
        }
    }

    Cpy::wait(barrier2);
    Cpt::compute(&smem2.to_slice(), &mut acc, config);

    Cpt::to_output(&mut acc, &mut output.to_slice_mut(), config);
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[allow(unused)]
/// Strategy in use. Refer to the struct of the same name as the variant for doc.
enum CopyStrategyEnum {
    DummyCopy,
    CoalescedCopy,
    MemcpyAsyncSingleSliceDuplicatedAll,
    MemcpyAsyncSingleSliceElected,
    MemcpyAsyncSingleSliceElectedCooperative,
    MemcpyAsyncSplitPlaneDuplicatedUnit,
    MemcpyAsyncSplitPlaneElectedUnit,
    MemcpyAsyncSplitDuplicatedAll,
    MemcpyAsyncSplitLargeUnitWithIdle,
    MemcpyAsyncSplitSmallUnitCoalescedLoop,
    MemcpyAsyncSplitMediumUnitCoalescedOnce,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
#[allow(unused)]
enum ComputeTaskEnum {
    Dummy,
}

fn launch_ref<R: Runtime, E: Float>(
    strategy: CopyStrategyEnum,
    client: &ComputeClient<R::Server, R::Channel>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    smem_size: u32,
    double_buffering: bool,
) {
    let cube_count = CubeCount::Static(1, 1, 1);
    let plane_dim = 32;
    let num_planes = 8;
    let cube_dim = CubeDim::new_2d(plane_dim, num_planes);
    let config = Config {
        plane_dim,
        num_planes,
        smem_size,
        acc_len: smem_size / (plane_dim * num_planes),
        double_buffering,
    };

    unsafe {
        match strategy {
            CopyStrategyEnum::DummyCopy => {
                memcpy_test::launch_unchecked::<E, DummyCopy, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::CoalescedCopy => {
                memcpy_test::launch_unchecked::<E, CoalescedCopy, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSingleSliceDuplicatedAll => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSingleSliceDuplicatedAll,
                    DummyCompute,
                    R,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSingleSliceElected => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSingleSliceElected, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSingleSliceElectedCooperative => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSingleSliceElectedCooperative,
                    DummyCompute,
                    R,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitPlaneDuplicatedUnit => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSplitPlaneDuplicatedUnit,
                    DummyCompute,
                    R,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitPlaneElectedUnit => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSplitPlaneElectedUnit, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitDuplicatedAll => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSplitDuplicatedAll, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitLargeUnitWithIdle => {
                memcpy_test::launch_unchecked::<E, MemcpyAsyncSplitLargeUnitWithIdle, DummyCompute, R>(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitSmallUnitCoalescedLoop => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSplitSmallUnitCoalescedLoop,
                    DummyCompute,
                    R,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
            CopyStrategyEnum::MemcpyAsyncSplitMediumUnitCoalescedOnce => {
                memcpy_test::launch_unchecked::<
                    E,
                    MemcpyAsyncSplitMediumUnitCoalescedOnce,
                    DummyCompute,
                    R,
                >(
                    client,
                    cube_count,
                    cube_dim,
                    input.as_tensor_arg(1),
                    output.as_tensor_arg(1),
                    config,
                )
            }
        }
    }
}

impl<R: Runtime, E: Float> Benchmark for MemcpyAsyncBench<R, E> {
    type Args = (TensorHandle<R, E>, TensorHandle<R, E>);

    fn prepare(&self) -> Self::Args {
        let client = R::client(&self.device);

        (
            TensorHandle::zeros(&client, vec![self.data_count]),
            TensorHandle::zeros(&client, vec![self.window_size]),
        )
    }

    fn execute(&self, args: Self::Args) {
        let smem_size = args.1.shape[0] as u32;
        launch_ref::<R, E>(
            self.strategy,
            &self.client,
            &args.0.as_ref(),
            &args.1.as_ref(),
            smem_size,
            self.double_buffering,
        );
    }

    fn name(&self) -> String {
        let client = R::client(&self.device);

        format!(
            "memcpy_async-{}-{}-{:?}",
            R::name(&client),
            E::as_elem_native_unchecked(),
            self.strategy
        )
        .to_lowercase()
    }

    fn sync(&self) {
        future::block_on(self.client.sync())
    }

    fn profile(&self, args: Self::Args) -> cubecl::benchmark::ProfileDuration {
        self.client.profile(|| self.execute(args))
    }
}

#[allow(dead_code)]
struct MemcpyAsyncBench<R: Runtime, E> {
    data_count: usize,
    window_size: usize,
    strategy: CopyStrategyEnum,
    double_buffering: bool,
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    _e: PhantomData<E>,
}

#[allow(dead_code)]
fn run<R: Runtime, E: Float>(device: R::Device, strategy: CopyStrategyEnum) {
    let client = R::client(&device);

    let data_count = 10000000;
    let window_size = 2048;

    let bench = MemcpyAsyncBench::<R, E> {
        data_count,
        window_size,
        strategy,
        double_buffering: true,
        client: client.clone(),
        device: device.clone(),
        _e: PhantomData,
    };
    println!("Data count: {data_count:?}, strategy: {strategy:?}");
    println!("{}", bench.name());
    println!("{}", bench.run(TimingMethod::Full));
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), CopyStrategyEnum::DummyCopy);
        run::<cubecl::cuda::CudaRuntime, f32>(Default::default(), CopyStrategyEnum::CoalescedCopy);
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSingleSliceDuplicatedAll,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSingleSliceElected,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSingleSliceElectedCooperative,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSplitPlaneDuplicatedUnit,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSplitPlaneElectedUnit,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSplitDuplicatedAll,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSplitLargeUnitWithIdle,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSplitSmallUnitCoalescedLoop,
        );
        run::<cubecl::cuda::CudaRuntime, f32>(
            Default::default(),
            CopyStrategyEnum::MemcpyAsyncSplitMediumUnitCoalescedOnce,
        );
    }
}
