use cubecl::{comptime, prelude::*};

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
enum OperationKind {
    Exp,
    Log,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
struct Operation {
    kind: OperationKind,
    input_index: u32,
    output_index: u32,
}

#[cube(launch_unchecked)]
fn fusing<F: Float>(
    inputs: &Sequence<Array<F>>,
    outputs: &mut Sequence<Array<F>>,
    #[comptime] ops: Sequence<Operation>,
) {
    #[unroll]
    for index in 0..ops.len() {
        let op = comptime! { ops.index(index) };
        let input = inputs.index(op.input_index);
        let output = outputs.index_mut(op.output_index);

        match op.kind {
            OperationKind::Exp => output[ABSOLUTE_POS] = F::exp(input[ABSOLUTE_POS]),
            OperationKind::Log => output[ABSOLUTE_POS] = F::log(input[ABSOLUTE_POS]),
        }
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let vectorization = 4;
    let output_handle_1 = client
        .empty(input.len() * core::mem::size_of::<f32>())
        .expect("Failed to allocate memory");
    let output_handle_2 = client
        .empty(input.len() * core::mem::size_of::<f32>())
        .expect("Failed to allocate memory");
    let input_handle = client
        .create(f32::as_bytes(input))
        .expect("Failed to allocate memory");

    let mut ops = Sequence::new();
    let mut inputs = SequenceArg::new();
    let mut outputs = SequenceArg::new();

    unsafe {
        inputs.push(ArrayArg::from_raw_parts::<f32>(
            &input_handle,
            input.len(),
            vectorization as u8,
        ));
        outputs.push(ArrayArg::from_raw_parts::<f32>(
            &output_handle_1,
            input.len(),
            vectorization as u8,
        ));
        outputs.push(ArrayArg::from_raw_parts::<f32>(
            &output_handle_2,
            input.len(),
            vectorization as u8,
        ));

        ops.push(Operation {
            kind: OperationKind::Exp,
            input_index: 0,
            output_index: 0,
        });
        ops.push(Operation {
            kind: OperationKind::Log,
            input_index: 0,
            output_index: 1,
        });

        fusing::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32 / vectorization, 1, 1),
            inputs,
            outputs,
            ops,
        )
    };

    let bytes = client.read_one(output_handle_1.binding());
    let output_1 = f32::from_bytes(&bytes);

    println!("Output 1 => {output_1:?}");

    let bytes = client.read_one(output_handle_2.binding());
    let output_2 = f32::from_bytes(&bytes);
    println!("Output 2 => {output_2:?}");
}
