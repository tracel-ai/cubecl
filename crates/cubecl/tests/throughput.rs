use cubecl::{Runtime, TestRuntime};
use cubecl_ir::{ElemType, FloatKind};
use cubecl_std::throughput::peak_throughput;

#[test]
pub fn test_throughput() {
    let client = TestRuntime::client(&Default::default());

    let output = peak_throughput(
        &client,
        cubecl_runtime::throughput::ThroughputMode::Direct,
        ElemType::Float(FloatKind::F32),
    );

    println!("{:?}", output);
    println!("flops: {} GB/s", output.throughput_gb_s());
}
