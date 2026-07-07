fn main() {
    throughput::dispatch!(R => throughput::all::<R>(&Default::default()));
}
