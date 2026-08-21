fn main() {
    throughput::dispatch!(R => throughput::memory_write::<R>(&Default::default()));
}
