use cubecl::linalg::tensor::index_offset_with_layout;
use cubecl::{comptime, prelude::*};
use half::f16;

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
enum DType {
    F32,
    F16,
}

#[derive(CubeType, Clone, Debug, Hash, PartialEq, Eq)]
enum Operation {
    Exp {
        input: Position,
        output: Position,
        dtype: DType,
    },
    Log {
        input: Position,
        output: Position,
    },
    Add {
        lhs: Position,
        rhs: Position,
        out: Position,
    },
}

#[derive(CubeType, Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Position {
    Input(u32),
    Output(u32),
    Local(u32),
}

#[cube]
pub fn read_f32(
    inputs: &FusionArrays,
    locals: &FusionLocals,
    #[comptime] position: Position,
) -> Line<f32> {
    match position {
        Position::Input(index) => inputs.inputs_f32.index(index)[ABSOLUTE_POS],
        Position::Local(index) => *locals.inputs_f32.index(index),
        Position::Output(_index) => comptime![panic!("Invalid")],
    }
}

#[cube]
pub fn write_f32(
    outputs: &mut FusionArrays,
    locals: &mut FusionLocals,
    value: Line<f32>,
    #[comptime] position: Position,
) {
    match position {
        Position::Output(index) => outputs.inputs_f32.index_mut(index)[ABSOLUTE_POS] = value,
        Position::Local(index) => locals.inputs_f32.insert(index, value),
        Position::Input(_index) => comptime![panic!("Invalid")],
    }
}

#[cube]
pub fn read_f16(
    inputs: &FusionArrays,
    locals: &FusionLocals,
    #[comptime] position: Position,
) -> Line<f16> {
    match position {
        Position::Input(index) => inputs.inputs_f16.index(index)[ABSOLUTE_POS],
        Position::Local(index) => *locals.inputs_f16.index(index),
        Position::Output(_index) => comptime![panic!("Invalid")],
    }
}

#[cube]
pub fn write_f16(
    outputs: &mut FusionArrays,
    locals: &mut FusionLocals,
    value: Line<f16>,
    #[comptime] position: Position,
) {
    match position {
        Position::Output(index) => outputs.inputs_f16.index_mut(index)[ABSOLUTE_POS] = value,
        Position::Local(index) => locals.inputs_f16.insert(index, value),
        Position::Input(_index) => comptime![panic!("Invalid")],
    }
}

#[derive(CubeLaunch)]
pub struct FusionArrays {
    inputs_f32: Sequence<Tensor<Line<f32>>>,
    inputs_f16: Sequence<Tensor<Line<f16>>>,
}

#[derive(CubeType)]
pub struct FusionLocals {
    inputs_f32: Sequence<Line<f32>>,
    inputs_f16: Sequence<Line<f16>>,
}

#[derive(CubeType)]
pub struct FusionTensors<'a> {
    pub item_f32: Sequence<&'a Tensor<Line<f32>>>,
    pub item_f16: Sequence<&'a Tensor<Line<f16>>>,
}

#[derive(CubeType)]
pub struct FusionTensorsMut<'a> {
    pub item_f32: Sequence<&'a mut Tensor<Line<f32>>>,
    pub item_f16: Sequence<&'a mut Tensor<Line<f16>>>,
}

#[cube(launch_unchecked)]
fn fusing(inputs: &FusionArrays, outputs: &mut FusionArrays, #[comptime] ops: Sequence<Operation>) {
    let mut locals = FusionLocals {
        inputs_f32: Sequence::new(),
        inputs_f16: Sequence::new(),
    };

    #[unroll]
    for index in 0..ops.len() {
        let op = comptime! { ops.index(index).clone() };
        match op {
            Operation::Exp {
                input,
                output,
                dtype,
            } => match dtype {
                DType::F32 => {
                    let input = read_f32(inputs, &locals, input);
                    let result = Line::exp(input);

                    write_f32(outputs, &mut locals, result, output);
                }
                DType::F16 => {
                    let input = read_f16(inputs, &locals, input);
                    let result = Line::exp(input);

                    write_f16(outputs, &mut locals, result, output);
                }
            },
            Operation::Log { input, output } => {
                let input = read_f32(inputs, &locals, input);
                let result = Line::log(input);

                write_f32(outputs, &mut locals, result, output);
            }
            Operation::Add { lhs, rhs, out } => {
                let lhs = read_f32(inputs, &locals, lhs);
                let rhs = read_f32(inputs, &locals, rhs);
                let result = lhs + rhs;

                write_f32(outputs, &mut locals, result, out);
            }
        }
    }
}

pub fn launch<R: Runtime>(device: &R::Device) {
    let client = R::client(device);
    let input = &[-1., 0., 1., 5.];
    let vectorization = 1;
    let output_handle_1 = client.empty(input.len() * core::mem::size_of::<f32>());
    let output_handle_2 = client.empty(input.len() * core::mem::size_of::<f32>());
    let input_handle = client.create(f32::as_bytes(input));

    let mut ops = Sequence::new();
    let mut inputs = SequenceArg::new();
    let mut outputs = SequenceArg::new();

    let shape = &[input.len()];
    let strides = &[1];

    unsafe {
        inputs.push(TensorArg::from_raw_parts(
            &input_handle,
            strides,
            shape,
            vectorization as u8,
        ));
        outputs.push(TensorArg::from_raw_parts(
            &output_handle_1,
            strides,
            shape,
            vectorization as u8,
        ));
        outputs.push(TensorArg::from_raw_parts(
            &output_handle_2,
            strides,
            shape,
            vectorization as u8,
        ));

        ops.push(Operation::Exp {
            input: Position::Input(0),
            output: Position::Local(0),
            dtype: DType::F32,
        });
        ops.push(Operation::Add {
            lhs: Position::Input(0),
            rhs: Position::Local(0),
            out: Position::Local(1),
        });
        ops.push(Operation::Exp {
            input: Position::Local(0),
            output: Position::Output(0),
            dtype: DType::F32,
        });
        ops.push(Operation::Log {
            input: Position::Local(1),
            output: Position::Output(1),
        });

        fusing::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(input.len() as u32 / vectorization, 1, 1),
            FusionArraysLaunch {
                inputs_f32: inputs,
                inputs_f16: SequenceArg::new(),
                _phantom_runtime: std::marker::PhantomData,
                _phantom_a: std::marker::PhantomData,
            },
            FusionArraysLaunch {
                inputs_f32: outputs,
                inputs_f16: SequenceArg::new(),
                _phantom_runtime: std::marker::PhantomData,
                _phantom_a: std::marker::PhantomData,
            },
            ops,
        )
    };

    let bytes = client.read(output_handle_1.binding());
    let output_1 = f32::from_bytes(&bytes);

    println!("Output 1 => {output_1:?}");

    let bytes = client.read(output_handle_2.binding());
    let output_2 = f32::from_bytes(&bytes);
    println!("Output 2 => {output_2:?}");
}
