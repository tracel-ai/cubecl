fn main() {
    throughput::dispatch!(device => throughput::pressure(&device));
}
