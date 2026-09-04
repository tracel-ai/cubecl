fn main() {
    throughput::dispatch!(device => throughput::memory_write(&device));
}
