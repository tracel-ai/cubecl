fn main() {
    throughput::dispatch!(device => throughput::all(&device));
}
