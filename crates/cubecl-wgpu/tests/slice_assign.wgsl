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
fn slice_assign_kernel(
    @builtin(local_invocation_index) local_idx: u32,
) {
let _0 = local_idx == 0u;
if _0 {
var l_1_0: u32;
l_1_0 = info[1u];
let slice_1_0_offset = 2u;
let slice_1_0_length = min(l_1_0, 3u) - 2u;
let slice_1_0_ptr = &output_0_global;
var l_1_1: u32;
var l_1_2: bool;
var l_1_3: f32;
l_1_1 = info[0u];
l_1_2 = 0u < l_1_1;
l_1_3 = input_0_global[0u];
let _1 = select(0f, l_1_3, l_1_2);
var l_1_4: u32;
var l_1_5: bool;
l_1_4 = slice_1_0_length;
l_1_5 = 0u < l_1_4;
if l_1_5 {
(*slice_1_0_ptr)[0u + slice_1_0_offset] = _1;
}
}
}
