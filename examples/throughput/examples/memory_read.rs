fn main() {
    throughput::dispatch!(R => throughput::memory_read::<R>(&Default::default()));
}
