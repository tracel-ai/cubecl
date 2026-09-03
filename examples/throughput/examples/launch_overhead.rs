fn main() {
    throughput::dispatch!(device => throughput::launch_overhead(&device));
}
