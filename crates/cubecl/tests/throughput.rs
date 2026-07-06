use cubecl::{Runtime, TestRuntime};
use cubecl_ir::{ElemType, FloatKind};
use cubecl_runtime::throughput::{ThroughputKey, ThroughputMode};
use cubecl_std::throughput::peak_throughput;

#[test]
pub fn test_throughput_compute_direct() {
    let client = TestRuntime::client(&Default::default());

    let dtype = ElemType::Float(FloatKind::F32);
    let mode = ThroughputMode::ComputeDirect;

    let key = ThroughputKey { mode, dtype };

    let output = peak_throughput(&client, key);

    println!("{:?}", output);
    println!("flops {dtype}: {}", output.format(&key));
}

#[test]
pub fn test_throughput_compute_cmma() {
    let client = TestRuntime::client(&Default::default());

    let dtype = ElemType::Float(FloatKind::F32);
    let mode = ThroughputMode::ComputeCmma;

    let key = ThroughputKey { mode, dtype };

    let output = peak_throughput(&client, key);

    println!("{:?}", output);
    println!("flops {dtype}: {}", output.format(&key));
}

#[test]
pub fn test_throughput_memory() {
    let client = TestRuntime::client(&Default::default());

    let dtype = ElemType::Float(FloatKind::F32);
    let mode = ThroughputMode::Memory;

    let key = ThroughputKey { mode, dtype };

    let output = peak_throughput(&client, key);

    println!("{:?}", output);
    println!("flops {dtype}: {}", output.format(&key));
}
