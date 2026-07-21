use cubecl_common::{future, stream_id::StreamId};
use cubecl_core as cubecl;
use cubecl_core::{
    Runtime,
    ir::{ElemType, FloatKind},
    prelude::*,
    server::{Binding, GemmDescriptor, GemmMatrix, GroupedGemmDescriptor, Handle},
};
use cubecl_cuda::{CudaDevice, CudaRuntime};
use half::bf16;

#[cube(launch)]
fn copy_bf16(input: &[bf16], output: &mut [bf16]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    }
}

#[derive(Clone, Copy, Debug)]
struct Problem {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    lhs_t: bool,
    rhs_t: bool,
    lhs_broadcast: bool,
    rhs_broadcast: bool,
    padding: usize,
    batch_gap: usize,
    offset: usize,
}

struct Matrix {
    bits: Vec<u16>,
    rows: usize,
    cols: usize,
    batches: usize,
    transposed: bool,
    ld: usize,
    batch_stride: usize,
    offset: usize,
}

fn main() {
    correctness_matrix();
    grouped_correctness();
    cross_stream_ordering();
    overlapping_output_is_rejected();
    aliased_output_is_rejected();
    foreign_output_is_rejected();
    zero_k_is_rejected();
    prior_error_is_not_bypassed();
    println!("cuBLAS BF16 padded/offset/batched/grouped/multistream checks passed");
}

fn correctness_matrix() {
    for lhs_t in [false, true] {
        for rhs_t in [false, true] {
            for lhs_broadcast in [false, true] {
                for rhs_broadcast in [false, true] {
                    check(Problem {
                        m: 5,
                        n: 7,
                        k: 3,
                        batch: 3,
                        lhs_t,
                        rhs_t,
                        lhs_broadcast,
                        rhs_broadcast,
                        padding: 5,
                        batch_gap: 11,
                        offset: 13,
                    });
                }
            }
        }
    }
}

fn check(problem: Problem) {
    let client = CudaRuntime::client(&CudaDevice::default());
    let lhs = Matrix::new(
        problem.m,
        problem.k,
        if problem.lhs_broadcast {
            1
        } else {
            problem.batch
        },
        problem.lhs_t,
        problem.padding,
        problem.batch_gap,
        problem.offset,
        3,
    );
    let rhs = Matrix::new(
        problem.k,
        problem.n,
        if problem.rhs_broadcast {
            1
        } else {
            problem.batch
        },
        problem.rhs_t,
        problem.padding + 2,
        problem.batch_gap + 3,
        problem.offset + 5,
        7,
    );
    let out = Matrix::zeros(
        problem.m,
        problem.n,
        problem.batch,
        false,
        problem.padding + 4,
        problem.batch_gap + 7,
        problem.offset + 9,
    );

    let lhs_base = client.create_from_slice(bytemuck::cast_slice(&lhs.bits));
    let rhs_base = client.create_from_slice(bytemuck::cast_slice(&rhs.bits));
    let out_base = client.empty(out.bits.len() * 2);
    let lhs_view = view(&lhs_base, lhs.offset, lhs.bits.len());
    let rhs_view = view(&rhs_base, rhs.offset, rhs.bits.len());
    let out_view = view(&out_base, out.offset, out.bits.len());
    let descriptor = GemmDescriptor::new(
        matrix_arg(&lhs, lhs_view, problem.lhs_broadcast),
        matrix_arg(&rhs, rhs_view, problem.rhs_broadcast),
        matrix_arg(&out, out_view, false),
        problem.m as u32,
        problem.n as u32,
        problem.k as u32,
        problem.batch as u32,
        ElemType::Float(FloatKind::BF16),
    );
    client.gemm(descriptor);
    let bytes = client.read_one_unchecked(out_base);
    let actual = bytemuck::cast_slice::<u8, u16>(&bytes);

    for batch in 0..problem.batch {
        for row in 0..problem.m {
            for col in 0..problem.n {
                let expected = (0..problem.k)
                    .map(|inner| {
                        lhs.get(if problem.lhs_broadcast { 0 } else { batch }, row, inner)
                            * rhs.get(if problem.rhs_broadcast { 0 } else { batch }, inner, col)
                    })
                    .sum::<f32>();
                let index = out.index(batch, row, col);
                let actual = bf16::from_bits(actual[index]).to_f32();
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "{problem:?}, b={batch}, row={row}, col={col}: {actual} != {expected}"
                );
            }
        }
    }
}

fn grouped_correctness() {
    let client = CudaRuntime::client(&CudaDevice::default());
    let elem = ElemType::Float(FloatKind::BF16);
    if !client
        .features()
        .matmul
        .accelerated_grouped_gemm
        .contains(&elem)
    {
        return;
    }
    let problems = [(3, 5, 4), (7, 2, 3), (4, 6, 5)];
    let mut matrices = Vec::with_capacity(problems.len());
    let mut groups = Vec::with_capacity(problems.len());

    for (index, (m, n, k)) in problems.into_iter().enumerate() {
        let lhs = Matrix::new(m, k, 1, index % 2 == 0, 3 + index, 0, 7, index + 2);
        let rhs = Matrix::new(k, n, 1, index % 2 != 0, 5 + index, 0, 11, index + 5);
        let out = Matrix::zeros(m, n, 1, false, 2 + index, 0, 13);
        let lhs_base = client.create_from_slice(bytemuck::cast_slice(&lhs.bits));
        let rhs_base = client.create_from_slice(bytemuck::cast_slice(&rhs.bits));
        let out_base = client.empty(out.bits.len() * 2);
        groups.push(GemmDescriptor::new(
            matrix_arg(&lhs, view(&lhs_base, lhs.offset, lhs.bits.len()), false),
            matrix_arg(&rhs, view(&rhs_base, rhs.offset, rhs.bits.len()), false),
            matrix_arg(&out, view(&out_base, out.offset, out.bits.len()), false),
            m as u32,
            n as u32,
            k as u32,
            1,
            elem,
        ));
        matrices.push((lhs, rhs, out, lhs_base, rhs_base, out_base));
    }

    let descriptor = GroupedGemmDescriptor::new(groups);
    for _ in 0..16 {
        client.grouped_gemm(descriptor.clone());
    }
    for (lhs, rhs, out, _lhs_base, _rhs_base, out_base) in matrices {
        let bytes = client.read_one_unchecked(out_base);
        let actual = bytemuck::cast_slice::<u8, u16>(&bytes);
        for row in 0..out.rows {
            for col in 0..out.cols {
                let expected = (0..lhs.cols)
                    .map(|inner| lhs.get(0, row, inner) * rhs.get(0, inner, col))
                    .sum::<f32>();
                let actual = bf16::from_bits(actual[out.index(0, row, col)]).to_f32();
                assert!(
                    (actual - expected).abs() <= 0.06,
                    "grouped m={}, n={}, k={}, row={row}, col={col}: {actual} != {expected}",
                    out.rows,
                    out.cols,
                    lhs.cols
                );
            }
        }
    }
}

fn cross_stream_ordering() {
    let mut producer = CudaRuntime::client(&CudaDevice::default());
    let mut gemm = producer.clone();
    let mut consumer = producer.clone();
    unsafe {
        producer.set_stream(StreamId { value: 100 });
        gemm.set_stream(StreamId { value: 101 });
        consumer.set_stream(StreamId { value: 102 });
    }

    let problem = Problem {
        m: 8,
        n: 6,
        k: 4,
        batch: 2,
        lhs_t: false,
        rhs_t: true,
        lhs_broadcast: false,
        rhs_broadcast: true,
        padding: 0,
        batch_gap: 0,
        offset: 0,
    };
    let lhs = Matrix::new(8, 4, 2, false, 0, 0, 0, 3);
    let rhs = Matrix::new(4, 6, 1, true, 0, 0, 0, 7);
    let lhs_handle = producer.create_from_slice(bytemuck::cast_slice(&lhs.bits));
    let rhs_handle = producer.create_from_slice(bytemuck::cast_slice(&rhs.bits));
    let gemm_out = gemm.empty(problem.batch * problem.m * problem.n * 2);
    let descriptor = GemmDescriptor::new(
        matrix_arg(&lhs, lhs_handle.binding(), false),
        matrix_arg(&rhs, rhs_handle.binding(), true),
        GemmMatrix::new(
            gemm_out.clone().binding(),
            problem.n as u32,
            (problem.m * problem.n) as u64,
            false,
        ),
        problem.m as u32,
        problem.n as u32,
        problem.k as u32,
        problem.batch as u32,
        ElemType::Float(FloatKind::BF16),
    );
    gemm.gemm(descriptor);

    let consumed = consumer.empty(problem.batch * problem.m * problem.n * 2);
    copy_bf16::launch::<CudaRuntime>(
        &consumer,
        CubeCount::Static(1, 1, 1),
        CubeDim::new(&consumer, 128),
        unsafe { BufferArg::from_raw_parts(gemm_out, problem.batch * problem.m * problem.n) },
        unsafe {
            BufferArg::from_raw_parts(consumed.clone(), problem.batch * problem.m * problem.n)
        },
    );
    let bytes = consumer.read_one_unchecked(consumed);
    let actual = bytemuck::cast_slice::<u8, u16>(&bytes);
    for batch in 0..problem.batch {
        for row in 0..problem.m {
            for col in 0..problem.n {
                let expected = (0..problem.k)
                    .map(|inner| lhs.get(batch, row, inner) * rhs.get(0, inner, col))
                    .sum::<f32>();
                let index = (batch * problem.m + row) * problem.n + col;
                let actual = bf16::from_bits(actual[index]).to_f32();
                assert!((actual - expected).abs() <= 0.06);
            }
        }
    }
}

fn overlapping_output_is_rejected() {
    let mut client = CudaRuntime::client(&CudaDevice::default());
    unsafe { client.set_stream(StreamId { value: 103 }) };

    let m = 2;
    let n = 3;
    let k = 4;
    let batches = 2;
    let lhs = client.create_from_slice(bytemuck::cast_slice(&vec![
        bf16::ONE.to_bits();
        batches * m * k
    ]));
    let rhs = client.create_from_slice(bytemuck::cast_slice(&vec![
        bf16::ONE.to_bits();
        batches * k * n
    ]));
    let out = client.empty(batches * m * n * 2);
    let descriptor = GemmDescriptor::new(
        GemmMatrix::new(lhs.binding(), k as u32, (m * k) as u64, false),
        GemmMatrix::new(rhs.binding(), n as u32, (k * n) as u64, false),
        GemmMatrix::new(out.binding(), n as u32, (m * n - 1) as u64, false),
        m as u32,
        n as u32,
        k as u32,
        batches as u32,
        ElemType::Float(FloatKind::BF16),
    );
    client.gemm(descriptor);
    assert!(future::block_on(client.sync()).is_err());
}

fn aliased_output_is_rejected() {
    let mut client = CudaRuntime::client(&CudaDevice::default());
    unsafe { client.set_stream(StreamId { value: 104 }) };

    let values = vec![bf16::ONE.to_bits(); 4];
    let lhs_and_out = client.create_from_slice(bytemuck::cast_slice(&values));
    let rhs = client.create_from_slice(bytemuck::cast_slice(&values));
    client.gemm(GemmDescriptor::new(
        GemmMatrix::new(lhs_and_out.clone().binding(), 2, 0, false),
        GemmMatrix::new(rhs.binding(), 2, 0, false),
        GemmMatrix::new(lhs_and_out.binding(), 2, 0, false),
        2,
        2,
        2,
        1,
        ElemType::Float(FloatKind::BF16),
    ));
    assert!(future::block_on(client.sync()).is_err());
}

fn foreign_output_is_rejected() {
    let mut origin = CudaRuntime::client(&CudaDevice::default());
    let mut execution = origin.clone();
    unsafe {
        origin.set_stream(StreamId { value: 105 });
        execution.set_stream(StreamId { value: 106 });
    }

    let values = vec![bf16::ONE.to_bits(); 4];
    let lhs = execution.create_from_slice(bytemuck::cast_slice(&values));
    let rhs = execution.create_from_slice(bytemuck::cast_slice(&values));
    let foreign_out = origin.empty(8);
    execution.gemm(GemmDescriptor::new(
        GemmMatrix::new(lhs.binding(), 2, 0, false),
        GemmMatrix::new(rhs.binding(), 2, 0, false),
        GemmMatrix::new(foreign_out.binding(), 2, 0, false),
        2,
        2,
        2,
        1,
        ElemType::Float(FloatKind::BF16),
    ));
    assert!(future::block_on(execution.sync()).is_err());
}

fn zero_k_is_rejected() {
    let mut client = CudaRuntime::client(&CudaDevice::default());
    unsafe { client.set_stream(StreamId { value: 107 }) };

    let placeholder = client.empty(2);
    let out = client.empty(12);
    client.gemm(GemmDescriptor::new(
        GemmMatrix::new(placeholder.clone().binding(), 1, 0, false),
        GemmMatrix::new(placeholder.binding(), 3, 0, false),
        GemmMatrix::new(out.binding(), 3, 0, false),
        2,
        3,
        0,
        1,
        ElemType::Float(FloatKind::BF16),
    ));
    assert!(future::block_on(client.sync()).is_err());
}

fn prior_error_is_not_bypassed() {
    let client = CudaRuntime::client(&CudaDevice::default());
    let input = client.create_from_slice(bytemuck::cast_slice(&[bf16::ONE.to_bits()]));
    let output = client.empty(2);

    // The CUDA launch limit is at most 1024 units per cube. The fire-and-forget
    // launch records a server error; the following GEMM must observe that
    // unhealthy stream instead of enqueueing through it.
    copy_bf16::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(2048),
        unsafe { BufferArg::from_raw_parts(input.clone(), 1) },
        unsafe { BufferArg::from_raw_parts(output.clone(), 1) },
    );
    let descriptor = GemmDescriptor::new(
        GemmMatrix::new(input.clone().binding(), 1, 0, false),
        GemmMatrix::new(input.binding(), 1, 0, false),
        GemmMatrix::new(output.clone().binding(), 1, 0, false),
        1,
        1,
        1,
        1,
        ElemType::Float(FloatKind::BF16),
    );
    client.gemm(descriptor.clone());
    // Surfaces the original launch error and restores stream health.
    assert!(future::block_on(client.sync()).is_err());

    client.gemm(descriptor);
    let bytes = client.read_one_unchecked(output);
    assert_eq!(
        bytemuck::cast_slice::<u8, u16>(&bytes),
        &[bf16::ONE.to_bits()]
    );
}

impl Matrix {
    #[allow(clippy::too_many_arguments)]
    fn new(
        rows: usize,
        cols: usize,
        batches: usize,
        transposed: bool,
        padding: usize,
        batch_gap: usize,
        offset: usize,
        seed: usize,
    ) -> Self {
        let mut matrix = Self::zeros(rows, cols, batches, transposed, padding, batch_gap, offset);
        for batch in 0..batches {
            for row in 0..rows {
                for col in 0..cols {
                    let logical = (batch * rows + row) * cols + col;
                    let value = ((logical + seed) % 11) as f32 / 32.0 - 0.15;
                    let index = matrix.index(batch, row, col);
                    matrix.bits[index] = bf16::from_f32(value).to_bits();
                }
            }
        }
        matrix
    }

    #[allow(clippy::too_many_arguments)]
    fn zeros(
        rows: usize,
        cols: usize,
        batches: usize,
        transposed: bool,
        padding: usize,
        batch_gap: usize,
        offset: usize,
    ) -> Self {
        let ld = (if transposed { rows } else { cols }) + padding;
        let span = (if transposed { cols } else { rows }) * ld;
        let batch_stride = span + batch_gap;
        let suffix = 17;
        Self {
            bits: vec![0; offset + batches * batch_stride + suffix],
            rows,
            cols,
            batches,
            transposed,
            ld,
            batch_stride,
            offset,
        }
    }

    fn index(&self, batch: usize, row: usize, col: usize) -> usize {
        assert!(batch < self.batches && row < self.rows && col < self.cols);
        let matrix = self.offset + batch * self.batch_stride;
        if self.transposed {
            matrix + col * self.ld + row
        } else {
            matrix + row * self.ld + col
        }
    }

    fn get(&self, batch: usize, row: usize, col: usize) -> f32 {
        bf16::from_bits(self.bits[self.index(batch, row, col)]).to_f32()
    }
}

fn matrix_arg(matrix: &Matrix, binding: Binding, broadcast: bool) -> GemmMatrix {
    GemmMatrix::new(
        binding,
        matrix.ld as u32,
        if broadcast {
            0
        } else {
            matrix.batch_stride as u64
        },
        matrix.transposed,
    )
}

fn view(base: &Handle, offset: usize, total: usize) -> Binding {
    let suffix = total - offset;
    base.clone()
        .offset_start((offset * 2) as u64)
        .offset_end((suffix.min(17) * 2) as u64)
        .binding()
}
