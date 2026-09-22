fn main() {
    throughput::dispatch!(device => throughput::bounds(&device));
}
