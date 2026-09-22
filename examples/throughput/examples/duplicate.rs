fn main() {
    throughput::dispatch!(device => throughput::duplicate(&device));
}
