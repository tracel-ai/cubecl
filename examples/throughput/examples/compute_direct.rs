fn main() {
    throughput::dispatch!(R => throughput::compute_direct::<R>(&Default::default()));
}
