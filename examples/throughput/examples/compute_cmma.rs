fn main() {
    throughput::dispatch!(device => throughput::compute_cmma(&device));
}
