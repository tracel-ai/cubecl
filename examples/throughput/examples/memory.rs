fn main() {
    throughput::dispatch!(device => throughput::memory(&device));
}
