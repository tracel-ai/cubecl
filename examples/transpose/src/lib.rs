use cubecl::prelude::*;
use cubecl::server::Handle;

/// Naive transpose kernel - simple but not optimized for memory coalescing
#[cube(launch_unchecked)]
fn transpose_naive<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    rows: u32,
    cols: u32,
) {
    let pos = ABSOLUTE_POS;
    
    if pos < rows * cols {
        let row = pos / cols;
        let col = pos % cols;
        
        // Transpose: (row, col) -> (col, row)
        let out_pos = col * rows + row;
        output[out_pos] = input[pos];
    }
}

/// Trait for different transpose strategies
#[cube]
trait TransposeStrategy: 'static + Send + Sync {
    fn transpose<F: Float>(
        input: &Slice<F>,
        output: &mut SliceMut<F>,
        row: u32,
        col: u32,
        rows: u32,
        cols: u32,
    );
}

struct NaiveTranspose;

#[cube]
impl TransposeStrategy for NaiveTranspose {
    fn transpose<F: Float>(
        input: &Slice<F>,
        output: &mut SliceMut<F>,
        row: u32,
        col: u32,
        rows: u32,
        cols: u32,
    ) {
        let idx = row * cols + col;
        let out_idx = col * rows + row;
        output[out_idx] = input[idx];
    }
}

#[cube(launch_unchecked)]
fn transpose_trait<F: Float, S: TransposeStrategy>(
    input: &Array<F>,
    output: &mut Array<F>,
    rows: u32,
    cols: u32,
) {
    let pos = ABSOLUTE_POS;
    
    if pos < rows * cols {
        let row = pos / cols;
        let col = pos % cols;
        
        S::transpose(
            &input.to_slice(),
            &mut output.to_slice_mut(),
            row,
            col,
            rows,
            cols,
        );
    }
}

fn launch_naive<R: Runtime>(
    client: &ComputeClient<R::Server>,
    input: &Handle,
    output: &Handle,
    rows: usize,
    cols: usize,
) {
    let total_elements = rows * cols;
    
    unsafe {
        transpose_naive::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(total_elements as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(input, total_elements, 1),
            ArrayArg::from_raw_parts::<f32>(output, total_elements, 1),
            ScalarArg::new(rows as u32),
            ScalarArg::new(cols as u32),
        );
    }
}

fn launch_trait<R: Runtime, S: TransposeStrategy>(
    client: &ComputeClient<R::Server>,
    input: &Handle,
    output: &Handle,
    rows: usize,
    cols: usize,
) {
    let total_elements = rows * cols;
    
    unsafe {
        transpose_trait::launch_unchecked::<f32, S, R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(total_elements as u32, 1, 1),
            ArrayArg::from_raw_parts::<f32>(input, total_elements, 1),
            ArrayArg::from_raw_parts::<f32>(output, total_elements, 1),
            ScalarArg::new(rows as u32),
            ScalarArg::new(cols as u32),
        );
    }
}

#[derive(Debug)]
enum KernelKind {
    Naive,
    TraitNaive,
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    
    // Create a small test matrix for demonstration
    let rows = 4;
    let cols = 3;
    
    // Input matrix: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    let input_data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32).collect();
    
    println!("Input matrix ({}x{}):", rows, cols);
    for i in 0..rows {
        print!("  [");
        for j in 0..cols {
            print!("{:4}", input_data[i * cols + j]);
        }
        println!(" ]");
    }
    println!();
    
    let output = client.empty(input_data.len() * core::mem::size_of::<f32>());
    let input = client.create(f32::as_bytes(&input_data));
    
    for kind in [
        KernelKind::Naive,
        KernelKind::TraitNaive,
    ] {
        match kind {
            KernelKind::Naive => launch_naive::<R>(&client, &input, &output, rows, cols),
            KernelKind::TraitNaive => {
                launch_trait::<R, NaiveTranspose>(&client, &input, &output, rows, cols)
            }
        }
        
        let bytes = client.read_one(output.clone());
        let output_data = f32::from_bytes(&bytes);
        
        println!("[{:?} - {kind:?}]", R::name(&client));
        println!("Output matrix ({}x{}):", cols, rows);
        for i in 0..cols {
            print!("  [");
            for j in 0..rows {
                print!("{:4}", output_data[i * rows + j]);
            }
            println!(" ]");
        }
        
        // Verify correctness
        let mut correct = true;
        for i in 0..rows {
            for j in 0..cols {
                let original = input_data[i * cols + j];
                let transposed = output_data[j * rows + i];
                if (original - transposed).abs() > 1e-5 {
                    correct = false;
                    break;
                }
            }
        }
        
        if correct {
            println!("✓ Transpose correct\n");
        } else {
            println!("✗ Transpose incorrect\n");
        }
    }
}
