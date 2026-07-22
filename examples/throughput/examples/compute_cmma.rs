fn main() {
    throughput::dispatch!(R => throughput::compute_cmma::<R>(&Default::default()));
}
