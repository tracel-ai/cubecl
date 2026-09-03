fn main() {
    throughput::dispatch!(device => throughput::memory_curve(&device));
}
