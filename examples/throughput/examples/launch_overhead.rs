use cubecl::{Runtime, std::throughput::measure_launch_overhead};

fn main() {
    throughput::dispatch!(R =>  {
        let client = R::client(&Default::default());
        let duration = measure_launch_overhead::<R>(&client);
        println!("Launch overhead: {:?}", duration);
    });
}
