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

@compute
@workgroup_size(1, 1, 1)
fn main(
    @builtin(local_invocation_index) local_idx: u32,
) {let rank: u32 = info[0];
let _0 = local_idx == 0u;
if _0 {
let slice_1_0_offset = 2u;
let slice_1_0_length = 3u - 2u;
let slice_1_0_ptr = &output_0_global;
let _1 = input_0_global[0u];
(*slice_1_0_ptr)[0u + slice_1_0_offset] = _1;
}
}