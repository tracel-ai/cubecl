fn main() {
    throughput::dispatch!(R => throughput::memory_curve::<R>(&Default::default()));
}
