fn main() {
    throughput::dispatch!(device => throughput::contended(&device));
}
