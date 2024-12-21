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
let l_0_0 = local_idx == 0u;
if l_0_0 {
var l_mut_1_1: u32;
l_mut_1_1 = info[1u];
let slice_1_0_offset = 2u;
let slice_1_0_length = min(l_mut_1_1, 3u) - 2u;
let slice_1_0_ptr = &output_0_global;
var l_mut_1_2: u32;
var l_mut_1_3: bool;
var l_mut_1_4: f32;
l_mut_1_2 = info[0u];
l_mut_1_3 = 0u < l_mut_1_2;
l_mut_1_4 = input_0_global[0u];
let l_1_0 = select(0f, l_mut_1_4, l_mut_1_3);
var l_mut_1_5: u32;
var l_mut_1_6: bool;
l_mut_1_5 = slice_1_0_length;
l_mut_1_6 = 0u < l_mut_1_5;
if l_mut_1_6 {
(*slice_1_0_ptr)[0u + slice_1_0_offset] = l_1_0;
}
}
}