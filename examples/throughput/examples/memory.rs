fn main() {
    throughput::dispatch!(R => throughput::memory::<R>(&Default::default()));
}
