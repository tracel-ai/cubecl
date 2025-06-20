use cubecl_example::cpu_tensor::CpuTensor; // Change to the path of your own module containing the CpuTensor

/// This function execute the reduction in the following way by reducing the last dimension with a sum over each row a 2D matrix
/// [0 1 2]    [0 + 1 + 2]    [3 ]
/// [3 4 5] -> [3 + 4 + 5] -> [12]
/// [6 7 8]    [6 + 7 + 8]    [21]
fn reduce_matrix(input: &CpuTensor, output: &mut CpuTensor) {
    for i in 0..input.shape[0] {
        let mut acc = 0.0f32;
        for j in 0..input.shape[1] {
            acc += input.data[i * input.strides[0] + j];
        }
        output.data[i] = acc;
    }
}

fn launch() {
    let input_shape = vec![3, 3];
    let output_shape = vec![3];
    let input = CpuTensor::arange(input_shape);
    let mut output = CpuTensor::empty(output_shape);

    reduce_matrix(&input, &mut output);

    println!("Executed reduction => {:?}", output.read());
}

fn main() {
    launch();
}
