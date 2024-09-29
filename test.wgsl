@group(0)
@binding(0)
var<storage, read_write> input_0_global: array<f32>;

@group(0)
@binding(1)
var<storage, read_write> output_0_global: array<f32>;

@group(0)
@binding(2)
var<storage, read_write> info: array<u32>;

const WORKGROUP_SIZE_X = 1u;
const WORKGROUP_SIZE_Y = 1u;
const WORKGROUP_SIZE_Z = 1u;

var<workgroup> shared_0: array<f32, 1024>;

@compute
@workgroup_size(1, 1, 1)
fn main(
    @builtin(local_invocation_index) local_idx: u32,
) {let rank: u32 = info[0];
    let _0 = local_idx == 0u;
    if _0 {
        output_0_global[0] = dot(vec4(input_0_global[0]), vec4(input_0_global[1]));
    }
}