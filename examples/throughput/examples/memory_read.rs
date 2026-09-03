fn main() {
    throughput::dispatch!(device => throughput::memory_read(&device));
}
