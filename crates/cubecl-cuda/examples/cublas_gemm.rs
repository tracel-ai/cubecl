use cubecl_common::future;
use cubecl_core::{
    Runtime,
    ir::{ElemType, FloatKind},
    server::{GemmDescriptor, GemmMatrix},
};
use cubecl_cuda::{CudaDevice, CudaRuntime};
use half::bf16;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    lhs_t: bool,
    rhs_t: bool,
    lhs_broadcast: bool,
    rhs_broadcast: bool,
    repeats: usize,
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "check") {
        correctness_matrix();
        return;
    }
    if args.first().is_some_and(|arg| arg == "enqueue") {
        enqueue_overhead(args.get(1).map_or(10_000, |value| parse(value)));
        return;
    }
    if args.len() != 9 {
        eprintln!(
            "usage: cublas_gemm M N K BATCH LHS_T RHS_T LHS_BROADCAST RHS_BROADCAST REPEATS\n       cublas_gemm check\n       cublas_gemm enqueue [REPEATS]"
        );
        std::process::exit(2);
    }
    let shape = Shape {
        m: parse(&args[0]),
        n: parse(&args[1]),
        k: parse(&args[2]),
        batch: parse(&args[3]),
        lhs_t: parse_flag(&args[4]),
        rhs_t: parse_flag(&args[5]),
        lhs_broadcast: parse_flag(&args[6]),
        rhs_broadcast: parse_flag(&args[7]),
        repeats: parse(&args[8]),
    };
    let ms = benchmark(shape);
    let flops = 2.0 * shape.m as f64 * shape.n as f64 * shape.k as f64 * shape.batch as f64;
    println!(
        "m={} n={} k={} batch={} lhs_t={} rhs_t={} lhs_broadcast={} rhs_broadcast={} ms={ms:.4} tflops={:.2}",
        shape.m,
        shape.n,
        shape.k,
        shape.batch,
        shape.lhs_t,
        shape.rhs_t,
        shape.lhs_broadcast,
        shape.rhs_broadcast,
        flops / (ms * 1.0e9),
    );
}

fn enqueue_overhead(repeats: usize) {
    let client = CudaRuntime::client(&CudaDevice::default());
    let one = bf16::ONE.to_bits();
    let lhs = client.create_from_slice(bytemuck::cast_slice(&[one]));
    let rhs = client.create_from_slice(bytemuck::cast_slice(&[one]));
    let out = client.empty(2);
    let descriptor = descriptor(
        Shape {
            m: 1,
            n: 1,
            k: 1,
            batch: 1,
            lhs_t: false,
            rhs_t: false,
            lhs_broadcast: false,
            rhs_broadcast: false,
            repeats,
        },
        lhs.binding(),
        rhs.binding(),
        out.binding(),
    );
    client.gemm(descriptor.clone());
    future::block_on(client.sync()).unwrap();

    let start = Instant::now();
    for _ in 0..repeats {
        client.gemm(descriptor.clone());
    }
    let enqueue = start.elapsed();
    let drain_start = Instant::now();
    future::block_on(client.sync()).unwrap();
    let drain = drain_start.elapsed();
    println!(
        "repeats={repeats} enqueue_us_per_call={:.3} final_gpu_drain_ms={:.3}",
        enqueue.as_secs_f64() * 1.0e6 / repeats as f64,
        drain.as_secs_f64() * 1.0e3,
    );
}

fn benchmark(shape: Shape) -> f64 {
    let client = CudaRuntime::client(&CudaDevice::default());
    let lhs_batches = if shape.lhs_broadcast { 1 } else { shape.batch };
    let rhs_batches = if shape.rhs_broadcast { 1 } else { shape.batch };
    let lhs_elems = lhs_batches * shape.m * shape.k;
    let rhs_elems = rhs_batches * shape.k * shape.n;
    let out_elems = shape.batch * shape.m * shape.n;
    let lhs = client.create_from_slice(bytemuck::cast_slice(&values(lhs_elems, 3)));
    let rhs = client.create_from_slice(bytemuck::cast_slice(&values(rhs_elems, 7)));
    let out = client.empty(out_elems * 2);
    let descriptor = descriptor(shape, lhs.binding(), rhs.binding(), out.binding());

    for _ in 0..5 {
        client.gemm(descriptor.clone());
    }
    future::block_on(client.sync()).unwrap();

    let start = Instant::now();
    for _ in 0..shape.repeats {
        client.gemm(descriptor.clone());
    }
    future::block_on(client.sync()).unwrap();
    start.elapsed().as_secs_f64() * 1_000.0 / shape.repeats as f64
}

fn correctness_matrix() {
    for lhs_t in [false, true] {
        for rhs_t in [false, true] {
            for batch in [1, 3] {
                for lhs_broadcast in [false, true] {
                    for rhs_broadcast in [false, true] {
                        if batch == 1 && (lhs_broadcast || rhs_broadcast) {
                            continue;
                        }
                        check(Shape {
                            m: 5,
                            n: 7,
                            k: 3,
                            batch,
                            lhs_t,
                            rhs_t,
                            lhs_broadcast,
                            rhs_broadcast,
                            repeats: 1,
                        });
                    }
                }
            }
        }
    }
    println!("cuBLAS BF16 layout correctness matrix passed");
}

fn check(shape: Shape) {
    let client = CudaRuntime::client(&CudaDevice::default());
    let lhs_batches = if shape.lhs_broadcast { 1 } else { shape.batch };
    let rhs_batches = if shape.rhs_broadcast { 1 } else { shape.batch };
    let lhs_bits = values(lhs_batches * shape.m * shape.k, 3);
    let rhs_bits = values(rhs_batches * shape.k * shape.n, 7);
    let lhs = client.create_from_slice(bytemuck::cast_slice(&lhs_bits));
    let rhs = client.create_from_slice(bytemuck::cast_slice(&rhs_bits));
    let out = client.empty(shape.batch * shape.m * shape.n * 2);
    client.gemm(descriptor(
        shape,
        lhs.binding(),
        rhs.binding(),
        out.clone().binding(),
    ));
    let bytes = client.read_one_unchecked(out);
    let actual = bytemuck::cast_slice::<u8, u16>(&bytes);

    for batch in 0..shape.batch {
        for row in 0..shape.m {
            for col in 0..shape.n {
                let expected = (0..shape.k)
                    .map(|inner| {
                        get(
                            &lhs_bits,
                            shape.m,
                            shape.k,
                            shape.lhs_t,
                            if shape.lhs_broadcast { 0 } else { batch },
                            row,
                            inner,
                        ) * get(
                            &rhs_bits,
                            shape.k,
                            shape.n,
                            shape.rhs_t,
                            if shape.rhs_broadcast { 0 } else { batch },
                            inner,
                            col,
                        )
                    })
                    .sum::<f32>();
                let index = (batch * shape.m + row) * shape.n + col;
                let actual = bf16::from_bits(actual[index]).to_f32();
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "layout failed: lhs_t={} rhs_t={} lhs_broadcast={} rhs_broadcast={} batch={batch} row={row} col={col}: {actual} != {expected}",
                    shape.lhs_t,
                    shape.rhs_t,
                    shape.lhs_broadcast,
                    shape.rhs_broadcast,
                );
            }
        }
    }
}

fn descriptor(
    shape: Shape,
    lhs: cubecl_core::server::Binding,
    rhs: cubecl_core::server::Binding,
    out: cubecl_core::server::Binding,
) -> GemmDescriptor {
    GemmDescriptor::new(
        GemmMatrix::new(
            lhs,
            if shape.lhs_t { shape.m } else { shape.k } as u32,
            if shape.lhs_broadcast {
                0
            } else {
                (shape.m * shape.k) as u64
            },
            shape.lhs_t,
        ),
        GemmMatrix::new(
            rhs,
            if shape.rhs_t { shape.k } else { shape.n } as u32,
            if shape.rhs_broadcast {
                0
            } else {
                (shape.k * shape.n) as u64
            },
            shape.rhs_t,
        ),
        GemmMatrix::new(out, shape.n as u32, (shape.m * shape.n) as u64, false),
        shape.m as u32,
        shape.n as u32,
        shape.k as u32,
        shape.batch as u32,
        ElemType::Float(FloatKind::BF16),
    )
}

fn values(len: usize, offset: usize) -> Vec<u16> {
    (0..len)
        .map(|index| {
            let value = ((index + offset) % 11) as f32 / 32.0 - 0.15;
            bf16::from_f32(value).to_bits()
        })
        .collect()
}

fn get(
    bits: &[u16],
    rows: usize,
    cols: usize,
    transposed: bool,
    batch: usize,
    row: usize,
    col: usize,
) -> f32 {
    let base = batch * rows * cols;
    let index = if transposed {
        base + col * rows + row
    } else {
        base + row * cols + col
    };
    bf16::from_bits(bits[index]).to_f32()
}

fn parse(value: &str) -> usize {
    value.parse().unwrap()
}

fn parse_flag(value: &str) -> bool {
    match value {
        "0" => false,
        "1" => true,
        _ => panic!("flag must be 0 or 1"),
    }
}
