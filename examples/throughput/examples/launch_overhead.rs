fn main() {
    throughput::dispatch!(R => throughput::launch_overhead::<R>(&Default::default()));
}
