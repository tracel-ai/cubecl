use cubecl::{comptime, prelude::*};

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
enum OperationKind {
    Exp,
    Log,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
struct Operation {
    kind: OperationKind,
    input_index: usize,
    output_index: usize,
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
            OperationKind::Exp => output[ABSOLUTE_POS] = input[ABSOLUTE_POS].exp(),
            OperationKind::Log => output[ABSOLUTE_POS] = input[ABSOLUTE_POS].ln(),
        }
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let line_size = 4;
    let output_handle_1 = client.empty(input.len() * core::mem::size_of::<f32>());
    let output_handle_2 = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create_from_slice(f32::as_bytes(input));

    let mut ops = Sequence::new();
    let mut inputs = SequenceArg::new();
    let mut outputs = SequenceArg::new();

    unsafe {
        inputs.push(ArrayArg::from_raw_parts::<f32>(
            input_handle,
            input.len(),
            line_size,
        ));
        outputs.push(ArrayArg::from_raw_parts::<f32>(
            output_handle_1.clone(),
            input.len(),
            line_size,
        ));
        outputs.push(ArrayArg::from_raw_parts::<f32>(
            output_handle_2.clone(),
            input.len(),
            line_size,
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
            CubeDim::new_1d(input.len() as u32 / line_size as u32),
            inputs,
            outputs,
            ops,
        )
    };

    let bytes = client.read_one(output_handle_1).unwrap();
    let output_1 = f32::from_bytes(&bytes);

    println!("Output 1 => {output_1:?}");

    let bytes = client.read_one(output_handle_2).unwrap();
    let output_2 = f32::from_bytes(&bytes);
    println!("Output 2 => {output_2:?}");
}
