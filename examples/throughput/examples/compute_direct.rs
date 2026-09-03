fn main() {
    throughput::dispatch!(device => throughput::compute_direct(&device));
}
