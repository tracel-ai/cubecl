use cubecl_core::{
    frontend::{BufferArg, BufferBinding},
    ir::ElemType,
};
use cubecl_runtime::{
    client::ComputeClient,
    runtime::Runtime,
    server::{CubeCount, CubeDim},
    throughput::{ThroughputKey, ThroughputMode, ThroughputValue},
};

use crate::throughput::compute::compute_direct_throughput;

pub fn peak_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    mode: ThroughputMode,
    dtype: ElemType,
) -> ThroughputValue {
    let launch_info = match mode {
        ThroughputMode::Direct => direct_launch(client),
        _ => unimplemented!(),
        // ThroughputMode::TensorCore => tensor_core_launch(client),
    };

    let kernel = match mode {
        ThroughputMode::Direct => direct_throughput(client, dtype, &launch_info),
        _ => unimplemented!(),
        // ThroughputMode::TensorCore => tensor_core_throughput(client, dtype, launch_info),
    };

    let key = ThroughputKey { mode, dtype };

    client.throughput(key, launch_info.work_bytes(), kernel)
}

fn direct_launch<R: Runtime>(client: &ComputeClient<R>) -> LaunchInfo {
    let hardware = &client.properties().hardware;

    let plane = hardware.plane_size_max.max(1);
    let cube_dim = (hardware.max_units_per_cube.min(256) / plane * plane)
        .max(plane)
        .min(hardware.max_cube_dim.0);

    let sms = hardware.num_streaming_multiprocessors.unwrap_or(64);
    let cube_count = (sms * 32).min(hardware.max_cube_count.0);

    let n_acc = 8;
    let n_iter = 4096;

    LaunchInfo {
        cube_dim: cube_dim as usize,
        cube_count: cube_count as usize,
        n_iter,
        n_acc,
    }
}

fn direct_throughput<R: Runtime>(
    client: &ComputeClient<R>,
    dtype: ElemType,
    info: &LaunchInfo,
) -> impl Fn() {
    move || unsafe {
        let output_buffer = client.empty(1 * dtype.size());
        let output_buffer = BufferArg::Handle {
            handle: BufferBinding::from_raw_parts(output_buffer, 1),
        };

        compute_direct_throughput::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(info.cube_count as u32, 1, 1),
            CubeDim::new_1d(info.cube_dim as u32),
            output_buffer,
            info.n_acc as usize,
            info.n_iter as usize,
        )
    }
}

struct LaunchInfo {
    cube_dim: usize,
    cube_count: usize,
    n_iter: usize,
    n_acc: usize,
}

impl LaunchInfo {
    pub fn work_bytes(&self) -> usize {
        2usize * self.cube_count * self.cube_dim * self.n_iter * self.n_acc
    }
}
