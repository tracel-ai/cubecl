use cubecl::Device;
use cubecl_common::bytes::Bytes;
use std::time::{Duration, Instant};

fn main() {
    throughput::dispatch!(device => run(&device));
}

const ITERATIONS: usize = 10;

fn run(device: &Device) {
    let client = device.client();
    println!("Device: {}", client.name());

    for size_mib in [64usize, 256, 512] {
        let size = size_mib * 1024 * 1024;
        let host_data = vec![0xABu8; size];

        let warmup = client.create(Bytes::from_bytes_vec(host_data.clone()));
        let _ = client.read_one(warmup);

        let mut uploads = Vec::with_capacity(ITERATIONS);
        let mut reads = Vec::with_capacity(ITERATIONS);

        for _ in 0..ITERATIONS {
            let bytes = Bytes::from_bytes_vec(host_data.clone());

            let start = Instant::now();
            let handle = client.create(bytes);
            cubecl::future::block_on(client.sync()).unwrap();
            uploads.push(start.elapsed());

            let start = Instant::now();
            let read = client.read_one(handle).unwrap();
            reads.push(start.elapsed());

            // Catches a direct write that raced or landed at the wrong offset.
            assert!(read.iter().all(|&b| b == 0xAB), "readback mismatch");
        }

        println!(
            "{size_mib:>4} MiB  upload {}  read {}",
            summary(&mut uploads, size),
            summary(&mut reads, size),
        );
    }
}

fn summary(times: &mut [Duration], size: usize) -> String {
    times.sort();
    let median = times[times.len() / 2];
    let gib_s = size as f64 / (1u64 << 30) as f64 / median.as_secs_f64();
    format!(
        "median {:7.2} ms ({:5.2} GiB/s) min {:7.2} ms",
        median.as_secs_f64() * 1e3,
        gib_s,
        times[0].as_secs_f64() * 1e3
    )
}
